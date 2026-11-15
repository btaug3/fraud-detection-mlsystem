"""Model loading + scoring with a graceful-degradation ladder.

Tier 1  LightGBM pipeline            (primary)
Tier 2  Logistic-regression pipeline (separate artifact, no LightGBM dependency)
Tier 3  Heuristic rule               (pure Python, stored in metadata JSON)
Tier 4  Training fraud prior         (always available; never flags)
"""
import json
import logging
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from .compiled import CompiledScorer                                     # NEW

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
    fast: dict = field(default_factory=dict)                             # NEW: verified fast paths

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
        svc._compile()                                                   # NEW
        return svc

    def _compile(self):                                                  # NEW
        """Build numpy fast paths and VERIFY them against the canonical sklearn pipelines."""
        if os.getenv("FRAUD_DISABLE_FAST") == "1":                       # switch for before/after benchmarks
            return
        rng = np.random.default_rng(0)
        probe = rng.normal(0, 2, size=(512, len(C.RAW_FEATURES)))
        probe[:, 0] = rng.uniform(0, 172800, 512)
        probe[rng.random(probe.shape) < 0.02] = np.nan                   # include missing values
        probe[:, -1] = np.nan_to_num(np.abs(probe[:, -1]) * 50, nan=10.0)
        df = pd.DataFrame(probe, columns=C.RAW_FEATURES)
        for name, art in (("primary", self.primary), ("secondary", self.secondary)):
            if art is None:
                continue
            try:
                cs = CompiledScorer(art["pipeline"])
                diff = np.max(np.abs(cs.predict_proba(probe) - art["pipeline"].predict_proba(df)[:, 1]))
                if diff < 1e-9:
                    self.fast[name] = cs
                else:
                    log.warning("compiled %s disagrees with pipeline (max diff %.2e); using pipeline", name, diff)
            except Exception as e:
                log.warning("could not compile %s: %r", name, e)

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

    def score(self, R):
        """R: (n, 30) float matrix in RAW_FEATURES order; NaN = missing value."""
        R = np.asarray(R, dtype=float)
        forced = os.getenv("FRAUD_FORCE_FAILURE", "")
        for name, art in (("primary_lightgbm", self.primary), ("secondary_logreg", self.secondary)):
            key = name.split("_")[0]
            if art is None or key in forced:
                continue
            try:
                if key in self.fast:                                     # NEW: fast path when verified
                    p = self.fast[key].predict_proba(R)
                else:
                    p = art["pipeline"].predict_proba(pd.DataFrame(R, columns=C.RAW_FEATURES))[:, 1]
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