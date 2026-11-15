"""Batch scoring CLI.

Usage:
  python -m fraud.predict_batch --input data/sample/unseen_transactions.csv --output outputs/scored.csv

* validates schema; bad rows are quarantined to <output>_rejected.csv instead of crashing the job
* chunked, vectorised inference -> bounded memory on large files
* writes transaction_id, fraud_probability, is_fraud, served_by, model_version
* logs status + throughput; exit code 0 = ok, 2 = schema error, 1 = failure
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .serving.model_service import ModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("fraud.batch")

REQUIRED = [C.ID_COL, "Time", "Amount"]


def validate_chunk(df):
    """Return (clean_rows, rejected_rows_with_reason)."""
    df = df.copy()
    for c in C.RAW_FEATURES:
        if c not in df:
            df[c] = np.nan                                   # missing optional column -> imputed later
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    reason = pd.Series("", index=df.index)
    reason[df[C.ID_COL].isna()] += "missing_id;"
    reason[df["Time"].isna() | (df["Time"] < 0)] += "bad_time;"
    reason[df["Amount"].isna() | (df["Amount"] < 0) | (df["Amount"] > 1e6)] += "bad_amount;"
    reason[df[C.PCA_COLS].isna().sum(axis=1) > 14] += "too_many_missing;"
    bad = reason != ""
    return df.loc[~bad], df.loc[bad, [C.ID_COL]].assign(reject_reason=reason[bad])


def run(inp: Path, out: Path, chunksize: int = 50_000):
    t0 = time.perf_counter()
    svc = ModelService.load()
    log.info("model tier=%s version=%s load_errors=%s", svc.active_tier, svc.version, svc.load_errors)

    header = pd.read_csv(inp, nrows=0).columns
    missing_req = [c for c in REQUIRED if c not in header]
    if missing_req:
        raise SystemExit(f"input schema error: missing required columns {missing_req}")

    out.parent.mkdir(parents=True, exist_ok=True)
    rej_path = out.with_name(out.stem + "_rejected.csv")
    n_in = n_ok = n_rej = n_flag = 0
    t_score = 0.0
    tiers = {}
    for i, chunk in enumerate(pd.read_csv(inp, chunksize=chunksize, dtype={C.ID_COL: str})):
        n_in += len(chunk)
        clean, rej = validate_chunk(chunk)
        ts = time.perf_counter()
        res = svc.score(clean[C.RAW_FEATURES].to_numpy(float))
        t_score += time.perf_counter() - ts
        pd.DataFrame({C.ID_COL: clean[C.ID_COL].values,
                      "fraud_probability": res.probability.round(6),
                      "is_fraud": res.flagged.astype(int),
                      "served_by": res.tier,
                      "model_version": svc.version}).to_csv(out, mode="w" if i == 0 else "a",
                                                            header=(i == 0), index=False)
        rej.to_csv(rej_path, mode="w" if i == 0 else "a", header=(i == 0), index=False)
        n_ok += len(clean); n_rej += len(rej); n_flag += int(res.flagged.sum())
        tiers[res.tier] = tiers.get(res.tier, 0) + len(clean)
        log.info("chunk %d: in=%d scored=%d rejected=%d flagged=%d",
                 i, len(chunk), len(clean), len(rej), int(res.flagged.sum()))

    wall = time.perf_counter() - t0
    summary = {"records_in": n_in, "records_scored": n_ok, "records_rejected": n_rej,
               "records_flagged": n_flag, "served_by": tiers,
               "wall_seconds": round(wall, 3), "scoring_seconds": round(t_score, 3),
               "throughput_rps_end_to_end": round(n_in / wall, 1),
               "throughput_rps_scoring_only": round(n_ok / t_score, 1) if t_score else None}
    log.info("SUMMARY %s", json.dumps(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch fraud scoring")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--chunksize", type=int, default=50_000)
    a = ap.parse_args(argv)
    try:
        run(a.input, a.output, a.chunksize)
        return 0
    except SystemExit as e:
        log.error(str(e)); return 2
    except Exception:
        log.exception("batch job failed"); return 1


if __name__ == "__main__":
    sys.exit(main())