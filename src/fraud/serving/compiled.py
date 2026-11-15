"""Low-latency numpy scoring path compiled from a FITTED sklearn Pipeline.

Re-applies exactly the parameters learned on the training set (imputer medians,
clip bounds, optional scaler) with plain numpy, then calls the LightGBM booster or
the logistic coefficients directly. Verified against the pipeline at load time.
"""
import numpy as np

from ..config import PCA_COLS, RAW_FEATURES

_I_TIME = RAW_FEATURES.index("Time")
_I_AMT = RAW_FEATURES.index("Amount")
_I_PCA = [RAW_FEATURES.index(c) for c in PCA_COLS]


class CompiledScorer:
    def __init__(self, pipeline):
        pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
        steps = pre.named_steps
        self.medians = steps["impute"].statistics_.astype(float)
        self.lo, self.hi = steps["clip"].lower_, steps["clip"].upper_
        sc = steps.get("scale")
        self.mu = sc.mean_ if sc is not None else None
        self.sd = sc.scale_ if sc is not None else None
        if hasattr(clf, "booster_"):                         # LightGBM
            booster, it = clf.booster_, clf.best_iteration_ or None
            self._predict = lambda Z: booster.predict(Z, num_iteration=it)
        else:                                                # logistic regression
            w, b = clf.coef_.ravel(), float(clf.intercept_[0])
            self._predict = lambda Z: 1.0 / (1.0 + np.exp(-(Z @ w + b)))

    def features(self, R):
        R = np.asarray(R, dtype=float)
        hour = (R[:, _I_TIME] % 86400) / 3600.0
        Z = np.empty((R.shape[0], len(PCA_COLS) + 3))
        Z[:, :28] = R[:, _I_PCA]
        with np.errstate(invalid="ignore"):
            Z[:, 28] = np.log1p(np.maximum(R[:, _I_AMT], 0))
        Z[:, 29] = np.sin(2 * np.pi * hour / 24)
        Z[:, 30] = np.cos(2 * np.pi * hour / 24)
        nan = np.isnan(Z)
        if nan.any():                                        # impute with TRAIN medians
            Z[nan] = np.take(self.medians, np.nonzero(nan)[1])
        np.clip(Z, self.lo, self.hi, out=Z)                  # clip to TRAIN bounds
        if self.mu is not None:
            Z = (Z - self.mu) / self.sd
        return Z

    def predict_proba(self, R):
        return self._predict(self.features(R))