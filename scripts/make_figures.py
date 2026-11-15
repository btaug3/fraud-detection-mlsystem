"""Generate report figures from saved results (no retraining)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config as C                     # noqa: E402
from fraud.metrics import expected_cost           # noqa: E402

FIG = C.REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
test = pd.read_csv(C.REPORTS / "test_scores.csv")
val = pd.read_csv(C.REPORTS / "val_scores.csv")
meta = json.loads(C.METADATA_PATH.read_text())
thr = meta["threshold"]

# 1) Precision-recall curves on TEST
plt.figure(figsize=(6, 4.5))
for col, label in [("score_lgbm", "LightGBM"), ("score_lr", "Logistic regression")]:
    p, r, _ = precision_recall_curve(test.y, test[col])
    plt.plot(r, p, label=f"{label} (PR-AUC={average_precision_score(test.y, test[col]):.3f})")
plt.axhline(test.y.mean(), ls=":", c="grey", label=f"Random / prior ({test.y.mean():.4f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curve, test window")
plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(FIG / "pr_curve_test.png", dpi=150); plt.close()

# 2) Business cost vs threshold on VALIDATION (how the threshold was chosen)
grid = np.linspace(0.01, 0.95, 95)
costs = [expected_cost(val.y, (val.score_lgbm >= t).astype(int), val.amount) for t in grid]
plt.figure(figsize=(6, 4))
plt.plot(grid, costs)
plt.axvline(thr, ls="--", c="red", label=f"chosen threshold = {thr:.3f}")
plt.xlabel("Decision threshold"); plt.ylabel("Expected cost on validation (EUR)")
plt.title("Threshold tuning: FP = EUR 5, FN = amount + EUR 25")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "cost_vs_threshold_val.png", dpi=150); plt.close()

# 3) Feature importance (gain)
fi = pd.Series(meta["feature_importance_gain_top10"]).sort_values()
plt.figure(figsize=(6, 4))
plt.barh(fi.index, fi.values / fi.values.sum() * 100, color="steelblue")
plt.xlabel("Share of total gain in top 10 (%)"); plt.title("LightGBM feature importance (top 10)")
plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()

# 4) API latency before vs after fast path
b = json.loads((C.REPORTS / "api_benchmark_pipeline.json").read_text())
a = json.loads((C.REPORTS / "api_benchmark_compiled.json").read_text())
labels = ["p50", "p95", "p99"]
x = np.arange(3)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, title in [(axes[0], "sequential_server_handler", "Inside handler"),
                       (axes[1], "sequential_client", "Client round trip")]:
    ax.bar(x - 0.2, [b[key][k] for k in labels], 0.4, label="sklearn pipeline")
    ax.bar(x + 0.2, [a[key][k] for k in labels], 0.4, label="numpy fast path")
    ax.set_xticks(x, labels); ax.set_ylabel("ms"); ax.set_title(title)
axes[0].legend()
fig.suptitle("Single-transaction latency, 2,000 sequential requests")
plt.tight_layout(); plt.savefig(FIG / "latency_before_after.png", dpi=150); plt.close()

# 5) Batch throughput
bb = pd.DataFrame(json.loads((C.REPORTS / "batch_benchmark.json").read_text()))
bb = bb[bb.chunksize == 100_000]
plt.figure(figsize=(6, 4))
for scorer in ["pipeline", "compiled"]:
    d = bb[bb.scorer == scorer]
    plt.plot(d.records, d.rps_end_to_end, marker="o", label=f"{scorer}, end-to-end")
    plt.plot(d.records, d.rps_scoring_only, marker="s", ls="--", label=f"{scorer}, scoring only")
plt.xscale("log"); plt.xlabel("Records in file (log scale)"); plt.ylabel("Records / second")
plt.title("Batch CLI throughput (chunk size 100k)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "batch_throughput.png", dpi=150); plt.close()

# 6) Bootstrap 95% confidence interval for test PR-AUC (only 52 frauds -> uncertainty matters)
rng = np.random.default_rng(42)
idx = np.arange(len(test))
boot = []
for _ in range(1000):
    s = rng.choice(idx, len(idx), replace=True)
    if test.y.values[s].sum() > 0:
        boot.append(average_precision_score(test.y.values[s], test.score_lgbm.values[s]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Test PR-AUC LightGBM = {average_precision_score(test.y, test.score_lgbm):.4f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
boot_lr = []
for _ in range(1000):
    s = rng.choice(idx, len(idx), replace=True)
    if test.y.values[s].sum() > 0:
        boot_lr.append(average_precision_score(test.y.values[s], test.score_lr.values[s]))
lo2, hi2 = np.percentile(boot_lr, [2.5, 97.5])
print(f"Test PR-AUC LogReg   = {average_precision_score(test.y, test.score_lr):.4f}  95% bootstrap CI [{lo2:.3f}, {hi2:.3f}]")
print("Figures saved to", FIG)