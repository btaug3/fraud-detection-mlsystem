"""Feature and prediction drift: Population Stability Index (PSI) + two-sample KS test.

The reference profile (train decile bins + a 5,000-value sample per feature) is built
from the TRAINING partition and shipped alongside the model.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..config import MODEL_FEATURES
from ..features import FeatureBuilder

PSI_WARN, PSI_ALERT = 0.10, 0.25     # industry rule of thumb
KS_STAT_ALERT = 0.10                 # with large n, p-values are always tiny -> use effect size
MISSING_ALERT = 0.05                 # >5% missing in a field = upstream feed problem
SEASONAL = {"hour_sin", "hour_cos"}  # depend on WHEN the window was taken -> compare same-hour baselines instead
N_BINS, REF_SAMPLE = 10, 5000


def _edges(x):
    e = np.unique(np.nanquantile(x, np.linspace(0, 1, N_BINS + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e.tolist()


def _hist(x, edges):
    h, _ = np.histogram(x[~np.isnan(x)], bins=np.asarray(edges))
    return h / max(h.sum(), 1)


def psi(ref_frac, cur_frac, eps=1e-4):
    r, c = np.clip(ref_frac, eps, None), np.clip(cur_frac, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference_profile(train_raw, train_scores):
    feats = FeatureBuilder().transform(train_raw)
    rng = np.random.default_rng(0)
    prof = {"features": {}}
    for col in MODEL_FEATURES:
        x = feats[col].to_numpy(float)
        edges = _edges(x)
        prof["features"][col] = {"edges": edges, "frac": _hist(x, edges).tolist(),
                                 "sample": rng.choice(x, REF_SAMPLE, replace=False).round(5).tolist()}
    edges = _edges(train_scores)
    prof["score"] = {"edges": edges, "frac": _hist(train_scores, edges).tolist(),
                     "sample": rng.choice(train_scores, REF_SAMPLE, replace=False).round(6).tolist()}
    return prof


def drift_report(profile, current_raw, current_scores=None):
    raw = pd.DataFrame(current_raw)
    feats = FeatureBuilder().transform(raw)
    items = [(c, feats[c].to_numpy(float), profile["features"][c]) for c in MODEL_FEATURES]
    if current_scores is not None:
        items.append(("model_score", np.asarray(current_scores, float), profile["score"]))
    rows = []
    for name, x, ref in items:
        p = psi(np.asarray(ref["frac"]), _hist(x, ref["edges"]))
        ks = ks_2samp(np.asarray(ref["sample"]), x[~np.isnan(x)])
        miss = float(raw[name].isna().mean()) if name in raw else float(np.isnan(x).mean())  # missing BEFORE imputation
        if name in SEASONAL:
            status = "SEASONAL"
        elif miss > MISSING_ALERT or p >= PSI_ALERT:
            status = "ALERT"
        elif p >= PSI_WARN:
            status = "WARN"
        else:
            status = "OK"
        rows.append({"feature": name, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_drift": bool(ks.statistic >= KS_STAT_ALERT and ks.pvalue < 0.01),
                     "missing_rate": round(miss, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)