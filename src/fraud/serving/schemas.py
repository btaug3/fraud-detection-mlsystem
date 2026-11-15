"""Pydantic request/response schemas: the API contract.

Malformed payloads (wrong types, out-of-range values, NaN/inf, unknown fields,
missing required fields) are rejected with HTTP 422 before reaching the model.
"""
import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ..config import PCA_COLS

MAX_BATCH = 1000
MAX_AMOUNT = 1_000_000.0                 # sanity bound in EUR; above this = malformed
MAX_TIME = 10 * 365 * 86400


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
    transactions: list[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)  # type: ignore


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