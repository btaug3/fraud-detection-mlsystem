"""Imbalance-aware evaluation metrics and cost-based threshold selection."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

from .config import COST_FN_FIXED, COST_FP


def recall_at_precision(y, s, min_precision):
    """Best recall achievable while keeping precision >= min_precision."""
    p, r, _ = precision_recall_curve(y, s)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def expected_cost(y, yhat, amount):
    """Business cost in EUR: each false alarm costs COST_FP;
    each missed fraud costs its amount + COST_FN_FIXED."""
    y, yhat, amount = map(np.asarray, (y, yhat, amount))
    fp = (yhat == 1) & (y == 0)
    fn = (yhat == 0) & (y == 1)
    return float(fp.sum() * COST_FP + (amount[fn] + COST_FN_FIXED).sum())


def choose_threshold(y, s, amount):
    """Pick the decision threshold that minimises business cost - on VALIDATION data only."""
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99),
                                     np.quantile(s, np.linspace(0.99, 1, 200))]))
    costs = [expected_cost(y, (s >= t).astype(int), amount) for t in grid]
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in grid]
    i, j = int(np.argmin(costs)), int(np.argmax(f1s))
    return {"cost_threshold": float(grid[i]), "cost_at_threshold": float(costs[i]),
            "f1_threshold": float(grid[j]), "f1_at_threshold": float(f1s[j])}


def evaluate(y, s, amount, threshold):
    """Full metric set at a given threshold."""
    y, s, amount = np.asarray(y), np.asarray(s), np.asarray(amount)
    yhat = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_at_p80": recall_at_precision(y, s, 0.80),
        "recall_at_p90": recall_at_precision(y, s, 0.90),
        "threshold": float(threshold),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "cost_eur": expected_cost(y, yhat, amount),
        "no_model_cost_eur": float((amount[y == 1] + COST_FN_FIXED).sum()),
    }