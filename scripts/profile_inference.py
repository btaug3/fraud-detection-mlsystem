"""Break down single-transaction inference latency by pipeline stage."""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config as C  # noqa: E402

art = joblib.load(C.MODEL_PATH)
pipe = art["pipeline"]
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
fb, imp, clip = pre.named_steps["features"], pre.named_steps["impute"], pre.named_steps["clip"]

X = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv", nrows=1)[C.RAW_FEATURES]   # ONE transaction
F = fb.transform(X)
Z = pre.transform(X)


def timeit(fn, n=1000):
    fn()                                    # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000   # ms per call


rows = [
    ("Full sklearn pipeline (predict_proba)", timeit(lambda: pipe.predict_proba(X))),
    ("  1. FeatureBuilder (pandas)", timeit(lambda: fb.transform(X))),
    ("  2. SimpleImputer", timeit(lambda: imp.transform(F))),
    ("  3. QuantileClipper", timeit(lambda: clip.transform(F.to_numpy()))),
    ("  4. LightGBM via sklearn wrapper", timeit(lambda: clf.predict_proba(Z))),
    ("  4b. LightGBM raw booster (numpy)", timeit(lambda: clf.booster_.predict(Z, num_iteration=clf.best_iteration_))),
]
total = rows[0][1]
print(f"{'Stage':42s} {'ms/call':>9s} {'% of total':>11s}")
for name, ms in rows:
    print(f"{name:42s} {ms:9.3f} {100 * ms / total:10.1f}%")