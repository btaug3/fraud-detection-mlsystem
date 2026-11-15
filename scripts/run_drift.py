"""Build the reference profile from TRAIN, then measure drift on real and simulated windows."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config as C                                              # noqa: E402
from fraud.data import load_raw, temporal_split                            # noqa: E402
from fraud.monitoring.drift import build_reference_profile, drift_report   # noqa: E402
from fraud.serving.model_service import ModelService                       # noqa: E402

train, val, test, _ = temporal_split(load_raw())
svc = ModelService.load()
score = lambda df: svc.score(df[C.RAW_FEATURES].to_numpy(float)).probability

profile = build_reference_profile(train, score(train))
C.REFERENCE_PROFILE_PATH.write_text(json.dumps(profile))
print("reference profile saved ->", C.REFERENCE_PROFILE_PATH)

inflated = test.copy(); inflated["Amount"] *= 1.5
shifted = test.copy(); shifted["V14"] -= 1.0
missing = test.copy()
missing.loc[missing.sample(frac=0.3, random_state=0).index, ["V1", "V2", "V3"]] = np.nan
# (scenario name, data, feature that the scenario is designed to break)
scenarios = [("val_window (real)", val, None), ("test_window (real)", test, None),
             ("sim: Amount x1.5", inflated, "log_amount"), ("sim: V14 shifted -1", shifted, "V14"),
             ("sim: 30% missing V1-V3", missing, "V1")]

summary = []
for name, df, target in scenarios:
    s = score(df)
    rep = drift_report(profile, df, s)
    slug = name.split(" ")[0].replace(":", "")
    rep.to_csv(C.REPORTS / f"drift_{slug}_{len(summary)}.csv", index=False)
    feat = rep[(rep.status != "SEASONAL") & (rep.feature != "model_score")]
    t = rep[rep.feature == target].iloc[0] if target else None
    summary.append({
        "scenario": name,
        "alerts": int((feat.status == "ALERT").sum()),
        "warns": int((feat.status == "WARN").sum()),
        "top non-seasonal PSI": ", ".join(f"{r.feature}={r.psi:.2f}" for r in feat.head(3).itertuples()),
        "target feature": f"{target}: PSI={t.psi:.2f} miss={t.missing_rate:.0%} -> {t.status}" if target else "-",
        "hour PSI (seasonal)": round(float(rep[rep.status == "SEASONAL"].psi.max()), 2),
        "score PSI": float(rep.loc[rep.feature == "model_score", "psi"].iloc[0]),
        "flag rate": round(float((s >= svc.primary["threshold"]).mean()), 5)})
    if name.startswith("test_window"):
        top = rep[rep.feature != "model_score"].head(15).iloc[::-1]
        colours = {"ALERT": "#c0392b", "WARN": "#e67e22", "SEASONAL": "#2980b9", "OK": "#7f8c8d"}
        plt.figure(figsize=(8, 4.5))
        plt.barh(top.feature, top.psi, color=[colours[s_] for s_ in top.status])
        plt.axvline(0.10, ls="--", c="grey"); plt.axvline(0.25, ls="--", c="black")
        plt.xscale("log"); plt.xlabel("PSI (log scale)")
        plt.title("Feature PSI, train vs test window (blue = seasonal, red = alert)")
        plt.tight_layout(); plt.savefig(C.REPORTS / "figures" / "psi_test_window.png", dpi=150)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
out = pd.DataFrame(summary)
out.to_csv(C.REPORTS / "drift_summary.csv", index=False)
for row in summary:                       # print one block per scenario (easier to read than a wide table)
    print("\n" + row["scenario"])
    for k, v in row.items():
        if k != "scenario":
            print(f"   {k:22s} {v}")