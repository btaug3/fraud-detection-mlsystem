"""FastAPI online scoring service.
Run:  uvicorn fraud.serving.app:app --port 8000
"""
import json
import logging
import math
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ..config import PCA_COLS, RAW_FEATURES
from .model_service import ModelService

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fraud.api")

MAX_BATCH = 1000
MAX_AMOUNT = 1_000_000.0                 # sanity bound in EUR; above this = malformed
MAX_TIME = 10 * 365 * 86400


# ------------------------------ request / response schemas ------------------------------
class _TxnBase(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields -> 422 (catches upstream schema drift)
    transaction_id: str = Field(..., min_length=1, max_length=64)
    Time: float = Field(..., ge=0, le=MAX_TIME, description="seconds since stream start")
    Amount: float = Field(..., ge=0, le=MAX_AMOUNT)

    @field_validator("*", mode="after")
    @classmethod
    def _finite(cls, v):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


# V1..V28 optional: a missing PCA value is imputed with the training median
Transaction = create_model(
    "Transaction", __base__=_TxnBase,
    **{c: (Optional[float], Field(None, ge=-1e4, le=1e4)) for c in PCA_COLS},
)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision: str                     # BLOCK / REVIEW / APPROVE
    threshold: float
    served_by: str                    # which fallback tier produced the score
    model_version: str
    imputed_fields: list[str] = []


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


# ------------------------------ in-process telemetry ------------------------------
class Telemetry:
    def __init__(self, window=5000):
        self.lat = deque(maxlen=window)
        self.status = Counter()
        self.tiers = Counter()
        self.flagged = 0
        self.scored = 0
        self.started = time.time()

    def snapshot(self):
        lat = np.array(self.lat) if self.lat else np.array([0.0])
        total = sum(self.status.values()) or 1
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "requests_by_status": dict(self.status),
            "error_rate_5xx": round(sum(v for k, v in self.status.items() if k >= 500) / total, 5),
            "latency_ms": {f"p{q}": round(float(np.percentile(lat, q)), 3) for q in (50, 95, 99)},
            "records_by_tier": dict(self.tiers),
            "records_scored": self.scored,
            "flag_rate": round(self.flagged / self.scored, 5) if self.scored else 0.0,
        }


# ------------------------------ app ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = ModelService.load()
    app.state.tel = Telemetry()
    svc = app.state.svc
    log.info(json.dumps({"event": "startup", "tier": svc.active_tier, "version": svc.version,
                         "load_errors": svc.load_errors}))
    svc.score(np.zeros((1, len(RAW_FEATURES))))    # warm-up so the first real request is not slow
    yield


app = FastAPI(title="Fraud Scoring Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception:
        log.exception("unhandled error")
        resp, status = JSONResponse({"detail": "internal error"}, status_code=500), 500
    ms = (time.perf_counter() - t0) * 1000
    tel = request.app.state.tel
    tel.status[status] += 1
    if request.url.path.startswith("/predict"):
        tel.lat.append(ms)
    log.info(json.dumps({"event": "request", "path": request.url.path, "status": status, "ms": round(ms, 3)}))
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = [{"loc": ".".join(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "invalid payload", "errors": errs[:20]})


def _score(records):
    svc, tel = app.state.svc, app.state.tel
    rows = [r.model_dump() for r in records]
    R = np.array([[np.nan if r[c] is None else r[c] for c in RAW_FEATURES] for r in rows], dtype=float)
    res = svc.score(R)
    tel.tiers[res.tier] += len(rows)
    tel.scored += len(rows)
    tel.flagged += int(res.flagged.sum())
    degraded = res.tier != "primary_lightgbm"
    out = []
    for i, r in enumerate(rows):
        missing = [c for c in PCA_COLS if r[c] is None]
        flag = bool(res.flagged[i])
        decision = "BLOCK" if flag else ("REVIEW" if degraded or len(missing) > 5 else "APPROVE")
        out.append(Prediction(transaction_id=r["transaction_id"],
                              fraud_probability=round(float(res.probability[i]), 6),
                              is_fraud=flag, decision=decision, threshold=round(res.threshold, 6),
                              served_by=res.tier, model_version=svc.version, imputed_fields=missing))
    return out


@app.get("/health")
def health():
    svc = app.state.svc
    tier = svc.active_tier
    return {"status": "ok" if tier == "primary_lightgbm" else "degraded",
            "active_tier": tier, "model_version": svc.version, "load_errors": svc.load_errors}


@app.post("/predict", response_model=PredictResponse)
def predict(txn: Transaction):
    t0 = time.perf_counter()
    preds = _score([txn])
    return PredictResponse(predictions=preds, latency_ms=round((time.perf_counter() - t0) * 1000, 3))


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    preds = _score(req.transactions)
    return PredictResponse(predictions=preds, latency_ms=round((time.perf_counter() - t0) * 1000, 3))


@app.get("/metrics")
def metrics():
    return app.state.tel.snapshot()