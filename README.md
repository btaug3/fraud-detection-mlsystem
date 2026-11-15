# Fraud Detection: Mini Production ML System

End-to-end production ML system for real-time credit-card fraud detection (MLME assignment, Option B):
leakage-free temporal split → class-weighted LightGBM with cost-based threshold → FastAPI service with schema
validation and a 4-tier fallback ladder → batch scoring CLI → latency/throughput profiling → PSI/KS drift monitoring.

## Results (test window: 42,561 transactions, 52 frauds)

| Model | PR-AUC (95% CI) | ROC-AUC | Recall@P90 | F1 | Cost (EUR) |
|---|---|---|---|---|---|
| Dummy prior | 0.001 | 0.500 | 0.000 | 0.000 | 7,469 |
| Rule V17 < -2.563 | 0.401 | 0.760 | 0.000 | 0.621 | 4,672 |
| Logistic regression (balanced) | 0.683 [0.557, 0.832] | 0.978 | 0.635 | 0.628 | 3,125 |
| **LightGBM (scale_pos_weight=5)** | **0.735 [0.621, 0.842]** | **0.982** | **0.654** | **0.692** | **2,878** |

API: p50 / p99 handler latency **0.35 / 0.93 ms** (numpy fast path; 7.97 / 13.62 ms before).
Batch: **~74k records/s** end-to-end, 1M transactions in 13.5 s.

## Project layout
```
src/fraud/
  config.py            paths, feature lists, split fractions, cost model
  data.py              load, de-duplicate, strict temporal split
  features.py          leakage-safe preprocessing (fit on train only)
  metrics.py           PR-AUC, Recall@Precision, cost, threshold selection
  train.py             baselines + LightGBM sweep + artifact serialization
  predict_batch.py     batch scoring CLI
  serving/app.py       FastAPI service (/health, /predict, /predict/batch, /metrics)
  serving/model_service.py   model loading + 4-tier fallback ladder
  serving/compiled.py  verified numpy fast path
  monitoring/drift.py  PSI / KS / missing-rate drift checks
scripts/               benchmarks, profiling, drift scenarios, figures
tests/                 21 tests: leakage, edge cases, malformed input, fallback
artifacts/             trained models + metadata + drift reference profile
reports/               metrics, benchmark JSON, figures
```

## Setup (Windows PowerShell; Python 3.13)
```powershell
git clone https://github.com/btaug3/fraud-detection-mlsystem.git
cd fraud-detection-mlsystem
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"          # macOS/Linux: export PYTHONPATH=src
```
Data: download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it at `data/raw/creditcard.csv` (not committed: 144 MB).

## Run
| Task | Command |
|---|---|
| Train (baselines + LightGBM, writes `artifacts/`) | `python -m fraud.train` |
| Tests | `pytest -q` |
| Start API | `uvicorn fraud.serving.app:app --port 8000` → open http://127.0.0.1:8000/docs |
| Batch scoring | `python -m fraud.predict_batch --input data/sample/unseen_transactions.csv --output outputs/scored.csv` |
| API latency benchmark (server running) | `python scripts/benchmark_api.py --label compiled` |
| Inference profiling | `python scripts/profile_inference.py` |
| Batch throughput benchmark | `python scripts/benchmark_batch.py` |
| Drift scenarios | `python scripts/run_drift.py` |
| Fallback demo | `python scripts/check_fallback.py` |
| Report figures | `python scripts/make_figures.py` |

Pre-trained artifacts are committed, so the API, batch CLI and tests run without retraining
(`data/sample/unseen_transactions.csv` is produced by `train.py`; it is also committed).

## Example request
```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" `
  -d '{\"transaction_id\":\"t1\",\"Time\":160895,\"Amount\":127.14,\"V14\":-4.29,\"V17\":-8.71}'
```
Missing V-fields are imputed (listed in `imputed_fields`); unknown fields, negative amounts and
non-numeric values are rejected with HTTP 422.

## Fallback ladder
LightGBM → logistic regression → V17 rule → training prior. Degraded tiers never auto-approve
(decision = REVIEW or BLOCK). Simulate with `$env:FRAUD_FORCE_FAILURE="primary"`.