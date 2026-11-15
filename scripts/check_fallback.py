"""Demonstrates the graceful-degradation ladder: knock out tiers one by one."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config as C                                  # noqa: E402
from fraud.serving.model_service import ModelService          # noqa: E402

R = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv", nrows=3)[C.RAW_FEATURES].to_numpy(float)
svc = ModelService.load()
print("active:", svc.active_tier, "| version:", svc.version)

for forced in ["", "primary", "primary,secondary", "primary,secondary,rule"]:
    os.environ["FRAUD_FORCE_FAILURE"] = forced
    r = svc.score(R)
    label = forced or "none"
    print(f"{label:28s} -> {r.tier:18s} probs={r.probability.round(5)}")