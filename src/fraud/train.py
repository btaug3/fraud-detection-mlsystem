"""Training pipeline - VERSION 2: baselines + LightGBM + serialized artifacts.
Usage:  python -m fraud.train
"""
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline

from . import config as C
from .data import load_raw, temporal_split
from .features import make_preprocessor
from .metrics import choose_threshold, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("fraud.train")


def fit_rule(train):
    """Heuristic baseline searched on TRAIN only; reused later as the API's no-ML fallback."""
    best = {"f1": -1}
    for feat in ["V14", "V17", "V12", "V10"]:
        for t in np.quantile(train[feat], np.linspace(0.0005, 0.02, 60)):
            f1 = f1_score(train[C.TARGET], (train[feat] < t).astype(int), zero_division=0)
            if f1 > best["f1"]:
                best = {"feature": feat, "op": "<", "threshold": float(t), "f1": float(f1)}
    log.info("Rule: flag if %s < %.3f (train F1=%.3f)", best["feature"], best["threshold"], best["f1"])
    return best


def main():
    t0 = time.time()
    C.ARTIFACTS.mkdir(exist_ok=True)
    C.REPORTS.mkdir(exist_ok=True)
    train, val, test, cutoffs = temporal_split(load_raw())
    Xtr, ytr = train[C.RAW_FEATURES], train[C.TARGET].values
    Xva, yva = val[C.RAW_FEATURES], val[C.TARGET].values
    Xte, yte = test[C.RAW_FEATURES], test[C.TARGET].values
    neg_pos = float((ytr == 0).sum() / (ytr == 1).sum())
    results = {}

    # ---------------- Baseline 0: dummy ----------------
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    results["dummy_prior"] = {"test": evaluate(yte, dummy.predict_proba(Xte)[:, 1], test["Amount"], 0.5)}

    # ---------------- Baseline 1: heuristic rule ----------------
    rule = fit_rule(train)
    s_rule = (test[rule["feature"]] < rule["threshold"]).astype(float).values
    results["heuristic_rule"] = {"rule": rule, "test": evaluate(yte, s_rule, test["Amount"], 0.5)}

    # ---------------- Baseline 2: weighted logistic regression ----------------
    pre_lr = make_preprocessor(scale=True).fit(Xtr)
    lr = LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)
    lr.fit(pre_lr.transform(Xtr), ytr)
    baseline = Pipeline([("pre", pre_lr), ("clf", lr)])
    s_va_lr = baseline.predict_proba(Xva)[:, 1]
    thr_lr = choose_threshold(yva, s_va_lr, val["Amount"])
    s_te_lr = baseline.predict_proba(Xte)[:, 1]
    results["logreg_balanced"] = {
        "val": evaluate(yva, s_va_lr, val["Amount"], thr_lr["cost_threshold"]),
        "test": evaluate(yte, s_te_lr, test["Amount"], thr_lr["cost_threshold"]),
        "threshold_search": thr_lr,
    }

    # ---------------- Main model: LightGBM with class-weighted loss ----------------
    pre = make_preprocessor(scale=False).fit(Xtr)          # trees need no scaling; fitted on TRAIN
    Ztr, Zva = pre.transform(Xtr), pre.transform(Xva)
    sweep, best = [], None
    for spw in [1.0, 5.0, 25.0, round(np.sqrt(neg_pos), 1), round(neg_pos, 1)]:
        clf = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            scale_pos_weight=spw, random_state=C.RANDOM_STATE, n_jobs=4, verbose=-1)
        clf.fit(Ztr, ytr, eval_X=(Zva,), eval_y=(yva,), eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        ap = average_precision_score(yva, clf.predict_proba(Zva)[:, 1])
        sweep.append({"scale_pos_weight": spw, "best_iter": int(clf.best_iteration_), "val_pr_auc": float(ap)})
        log.info("LightGBM scale_pos_weight=%-7s iters=%4d  val PR-AUC=%.4f", spw, clf.best_iteration_, ap)
        if best is None or ap > best[0]:
            best = (ap, spw, clf)
    _, best_spw, clf = best
    model = Pipeline([("pre", pre), ("clf", clf)])
    s_va = model.predict_proba(Xva)[:, 1]
    thr = choose_threshold(yva, s_va, val["Amount"])          # chosen on VALIDATION
    threshold = thr["cost_threshold"]
    s_te = model.predict_proba(Xte)[:, 1]                     # TEST used once, here
    results["lightgbm"] = {
        "val": evaluate(yva, s_va, val["Amount"], threshold),
        "test": evaluate(yte, s_te, test["Amount"], threshold),
        "threshold_search": thr, "spw_sweep": sweep, "chosen_scale_pos_weight": best_spw,
    }

    for name, r in results.items():
        t = r["test"]
        log.info("%-16s TEST PR-AUC=%.4f ROC-AUC=%.4f R@P90=%.3f P=%.3f R=%.3f F1=%.3f cost=EUR %.0f",
                 name, t["pr_auc"], t["roc_auc"], t["recall_at_p90"], t["precision"],
                 t["recall"], t["f1"], t["cost_eur"])
    log.info("Chosen threshold (min validation cost) = %.4f", threshold)

    # ---------------- Serialize production artifacts ----------------
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    joblib.dump({"pipeline": model, "threshold": threshold, "version": version}, C.MODEL_PATH)
    joblib.dump({"pipeline": baseline, "threshold": thr_lr["cost_threshold"], "version": version}, C.BASELINE_PATH)
    fi = pd.Series(clf.booster_.feature_importance("gain"), index=C.MODEL_FEATURES).sort_values(ascending=False)
    metadata = {
        "model_version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_features": C.RAW_FEATURES, "model_features": C.MODEL_FEATURES,
        "threshold": threshold, "baseline_threshold": thr_lr["cost_threshold"],
        "fallback_rule": rule, "train_fraud_prior": float(ytr.mean()),
        "split": {**cutoffs, "n_train": len(train), "n_val": len(val), "n_test": len(test),
                  "fraud_train": int(ytr.sum()), "fraud_val": int(yva.sum()), "fraud_test": int(yte.sum())},
        "cost_model": {"cost_fp_eur": C.COST_FP, "cost_fn_fixed_eur": C.COST_FN_FIXED},
        "feature_importance_gain_top10": {k: float(v) for k, v in fi.head(10).items()},
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "lightgbm": lgb.__version__},
    }
    C.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    (C.REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))

    # scored partitions for report figures
    pd.DataFrame({"transaction_id": test[C.ID_COL], "y": yte, "amount": test["Amount"].values,
                  "score_lgbm": s_te, "score_lr": s_te_lr}).to_csv(C.REPORTS / "test_scores.csv", index=False)
    pd.DataFrame({"y": yva, "amount": val["Amount"].values, "score_lgbm": s_va}).to_csv(
        C.REPORTS / "val_scores.csv", index=False)

    # an 'unseen' input file for the batch CLI (labels removed)
    C.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    test[[C.ID_COL, *C.RAW_FEATURES]].to_csv(C.DATA_SAMPLE / "unseen_transactions.csv", index=False)
    log.info("Done in %.1fs. Artifacts written to %s", time.time() - t0, C.ARTIFACTS)


if __name__ == "__main__":
    main()