"""Batch CLI throughput (records/second) across input sizes, chunk sizes and scorers."""
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud import config as C            # noqa: E402
from fraud.predict_batch import run      # noqa: E402

logging.disable(logging.INFO)            # keep output clean
TMP = ROOT / "outputs" / "bench"
TMP.mkdir(parents=True, exist_ok=True)


def make_input(n):
    """Replicate the unseen test file up to n rows (new IDs)."""
    p = TMP / f"input_{n}.csv"
    if not p.exists():
        base = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv")
        reps = -(-n // len(base))
        df = pd.concat([base] * reps, ignore_index=True).iloc[:n]
        df[C.ID_COL] = [f"bench_{i:08d}" for i in range(n)]
        df.to_csv(p, index=False)
    return p


results = []
for n in (42_561, 250_000, 1_000_000):
    inp = make_input(n)
    for chunk in (10_000, 100_000):
        for mode in ("pipeline", "compiled"):
            os.environ["FRAUD_DISABLE_FAST"] = "1" if mode == "pipeline" else "0"
            s = run(inp, TMP / "out.csv", chunk)
            row = {"records": n, "chunksize": chunk, "scorer": mode,
                   "wall_s": s["wall_seconds"], "rps_end_to_end": s["throughput_rps_end_to_end"],
                   "rps_scoring_only": s["throughput_rps_scoring_only"]}
            print(row)
            results.append(row)

(C.REPORTS / "batch_benchmark.json").write_text(json.dumps(results, indent=2))
print("saved ->", C.REPORTS / "batch_benchmark.json")