"""Leakage, API contract, robustness and fallback tests.   Run: pytest -v"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud import config as C
from fraud.data import load_raw, temporal_split
from fraud.features import make_preprocessor
from fraud.serving.app import app

SAMPLE = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv", nrows=50)
needs_raw = pytest.mark.skipif(not C.DATA_RAW.exists(),
                               reason="data/raw/creditcard.csv not downloaded (see README)")


def payload(i=0, **override):
    row = SAMPLE.iloc[i]
    d = {"transaction_id": str(row[C.ID_COL]), **{c: float(row[c]) for c in C.RAW_FEATURES}}
    d.update(override)
    return d


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the startup (model loading) like a real server
        yield c


# ------------------------------ 1. leakage safeguards ------------------------------


@needs_raw
def test_temporal_split_has_no_overlap():
    tr, va, te, _ = temporal_split(load_raw())
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
    assert set(tr.transaction_id).isdisjoint(te.transaction_id)

@needs_raw
def test_preprocessor_learns_from_train_only():
    tr, _, te, _ = temporal_split(load_raw())
    pre = make_preprocessor(scale=True).fit(tr[C.RAW_FEATURES])
    pre_leaky = make_preprocessor(scale=True).fit(pd.concat([tr, te])[C.RAW_FEATURES])
    # if test data were used in fitting, the learned means would differ
    assert not np.allclose(pre.named_steps["scale"].mean_, pre_leaky.named_steps["scale"].mean_)
    before = pre.named_steps["scale"].mean_.copy()
    pre.transform(te[C.RAW_FEATURES])                  # transform must NOT refit
    assert np.array_equal(before, pre.named_steps["scale"].mean_)


# ------------------------------ 2. API happy path ------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["active_tier"] == "primary_lightgbm"


def test_predict_ok(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert 0 <= p["fraud_probability"] <= 1 and p["served_by"] == "primary_lightgbm"


# ------------------------------ 3. edge cases: missing values / outliers ------------------------------
def test_missing_values_are_imputed(client):
    d = payload()
    for c in ["V1", "V2", "V3"]:
        d.pop(c)                     # field absent
    d["V4"] = None                   # field explicitly null
    r = client.post("/predict", json=d)
    assert r.status_code == 200
    assert set(r.json()["predictions"][0]["imputed_fields"]) == {"V1", "V2", "V3", "V4"}


def test_extreme_outlier_is_clipped_not_crashing(client):
    r = client.post("/predict", json=payload(V14=-9999.0, Amount=999_999.0))
    assert r.status_code == 200
    assert np.isfinite(r.json()["predictions"][0]["fraud_probability"])


# ------------------------------ 4. malformed inputs -> clean 422 ------------------------------
@pytest.mark.parametrize("bad", [
    {"Amount": -5}, {"Amount": "abc"}, {"Time": -1}, {"V1": "NaN"}, {"V1": 1e9},
    {"unknown_field": 1}, {"transaction_id": ""},
])
def test_malformed_payloads_rejected_422(client, bad):
    r = client.post("/predict", json=payload(**bad))
    assert r.status_code == 422 and r.json()["detail"] == "invalid payload"


def test_missing_required_field_rejected(client):
    d = payload(); d.pop("Amount")
    assert client.post("/predict", json=d).status_code == 422


def test_non_json_body_rejected(client):
    r = client.post("/predict", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_endpoint_and_limits(client):
    r = client.post("/predict/batch", json={"transactions": [payload(i) for i in range(20)]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 20
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


# ------------------------------ 5. fallback / graceful degradation ------------------------------
@pytest.mark.parametrize("forced,tier", [
    ("primary", "secondary_logreg"),
    ("primary,secondary", "heuristic_rule"),
    ("primary,secondary,rule", "prior"),
])
def test_degradation_ladder(client, monkeypatch, forced, tier):
    monkeypatch.setenv("FRAUD_FORCE_FAILURE", forced)
    p = client.post("/predict", json=payload()).json()["predictions"][0]
    assert p["served_by"] == tier
    assert p["decision"] in {"REVIEW", "BLOCK"}       # degraded tiers never auto-approve


# ------------------------------ 6. batch CLI quarantines bad rows ------------------------------
def test_batch_cli_quarantines_bad_rows(tmp_path):
    from fraud.predict_batch import run
    df = SAMPLE.copy()
    df["Time"] = df["Time"].astype(object)      # allow a string to be inserted (pandas 3)
    df.loc[0, "Amount"] = -1                    # bad amount
    df.loc[1, "Time"] = "oops"                  # non-numeric time
    df.loc[2, C.PCA_COLS[:20]] = np.nan         # too many missing values
    df.loc[3, "V5"] = np.nan                    # one missing value -> imputed, still scored
    inp = tmp_path / "in.csv"
    df.to_csv(inp, index=False)
    s = run(inp, tmp_path / "out.csv", 20)
    assert s["records_rejected"] == 3 and s["records_scored"] == len(df) - 3
    out = pd.read_csv(tmp_path / "out.csv")
    assert out.fraud_probability.between(0, 1).all()


# ------------------------------ 7. fast path == canonical pipeline ------------------------------
def test_compiled_fast_path_matches_pipeline_on_real_data():
    from fraud.serving.model_service import ModelService
    svc = ModelService.load()
    assert set(svc.fast) == {"primary", "secondary"}
    df = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv", nrows=5000)[C.RAW_FEATURES]
    df.iloc[::7, 3] = np.nan                       # inject missing values
    R = df.to_numpy(float)
    for key, art in (("primary", svc.primary), ("secondary", svc.secondary)):
        ref = art["pipeline"].predict_proba(df)[:, 1]
        assert np.max(np.abs(svc.fast[key].predict_proba(R) - ref)) < 1e-9