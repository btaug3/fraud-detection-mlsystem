"""Leakage-safe preprocessing.

Every transformer learns its parameters in fit() - called on the TRAIN partition only -
and merely applies them in transform() on validation, test and live serving data.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_FEATURES, PCA_COLS, RAW_FEATURES


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Feature engineering with NO learned state.
    - Raw Time is dropped (a counter that only grows; it would leak the split boundary).
    - Hour of day is kept as sin/cos so hour 23 and hour 0 are close together.
    - Amount is heavily skewed -> log1p.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=RAW_FEATURES)   # unknown cols dropped, missing cols -> NaN
        X = X.apply(pd.to_numeric, errors="coerce")          # bad strings -> NaN (then imputed)
        out = X[PCA_COLS].copy()
        out["log_amount"] = np.log1p(X["Amount"].clip(lower=0))
        hour = (X["Time"] % 86400) / 3600.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return out[MODEL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_FEATURES)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each feature to the 0.1% / 99.9% quantiles learned on TRAIN.
    Protects the model from extreme outliers at inference time."""
    def __init__(self, q_low=0.001, q_high=0.999):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.q_low, axis=0)    # learned from train only
        self.upper_ = np.nanquantile(X, self.q_high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_preprocessor(scale: bool) -> Pipeline:
    steps = [
        ("features", FeatureBuilder()),
        ("impute", SimpleImputer(strategy="median")),   # medians learned from train
        ("clip", QuantileClipper()),                    # bounds learned from train
    ]
    if scale:                                           # only the logistic baseline needs scaling
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)