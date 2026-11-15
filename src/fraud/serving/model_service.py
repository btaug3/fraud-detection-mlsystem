"""Model loading + scoring with a graceful-degradation ladder.

Tier 1  LightGBM pipeline            (primary)
Tier 2  Logistic-regression pipeline (separate artifact, no LightGBM dependency)
Tier 3  Heuristic rule               (pure Python, stored in metadata JSON)
Tier 4  Training fraud prior         (always available; never flags)

If a tier fails to load OR raises while scoring, the next tier is used, and the
response records which tier served it so downstream systems can route to review.
"""
import json
import logging
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .. import config as C

log = logging.getLogger("fraud.service")


@dataclass
class Scored:
    probability: np.ndarray
    flagged: np.ndarray
    tier: str
    threshold: float


@dataclass
class ModelService:
    primary: dict | None = None
    secondary: dict | None = None
    metadata: dict = field(default_factory=dict)
    load_errors: dict = field(default_factory=dict)

    @classmethod
    def load(cls):
        svc = cls()
        try:
            svc.metadata = json.loads(C.METADATA_PATH.read_text())
        except Exception as e:
            svc.load_errors["metadata"] = repr(e)
            svc.metadata = {"model_version": "unknown", "train_fraud_prior": 0.0017, "fallback_rule": None}
        for attr, path in (("primary", C.MODEL_PATH), ("secondary", C.BASELINE_PATH)):
            try:
                setattr(svc, attr, joblib.load(path))
            except Exception as e:
                svc.load_errors[attr] = repr(e)
                log.error("failed to load %s from %s: %r", attr, path, e)
        return svc

    @property
    def active_tier(self):
        if self.primary is not None:
            return "primary_lightgbm"
        if self.secondary is not None:
            return "secondary_logreg"
        if self.metadata.get("fallback_rule"):
            return "heuristic_rule"
        return "prior"

    @property
    def version(self):
        return (self.primary or self.secondary or {}).get("version", self.metadata.get("model_version", "unknown"))

    def score(self, R: np.ndarray) -> Scored:
        """R: (n, 30) float matrix in RAW_FEATURES order; NaN = missing value."""
        R = np.asarray(R, dtype=float)
        forced = os.getenv("FRAUD_FORCE_FAILURE", "")   # chaos switch for testing, e.g. "primary,secondary"
        for name, art in (("primary_lightgbm", self.primary), ("secondary_logreg", self.secondary)):
            if art is None or name.split("_")[0] in forced:
                continue
            try:
                X = pd.DataFrame(R, columns=C.RAW_FEATURES)
                p = art["pipeline"].predict_proba(X)[:, 1]
                if not np.all(np.isfinite(p)):
                    raise ValueError("non-finite score")
                return Scored(p, p >= art["threshold"], name, art["threshold"])
            except Exception as e:
                log.exception("tier %s failed, degrading: %r", name, e)
        rule = self.metadata.get("fallback_rule")
        if rule and "rule" not in forced:
            v = R[:, C.RAW_FEATURES.index(rule["feature"])]
            flag = np.nan_to_num(v, nan=np.inf) < rule["threshold"]
            return Scored(flag.astype(float), flag, "heuristic_rule", 0.5)
        prior = float(self.metadata.get("train_fraud_prior", 0.0017))
        return Scored(np.full(len(R), prior), np.zeros(len(R), bool), "prior", 1.0)