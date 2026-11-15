"""Empirical latency profiling of the running FastAPI service.

1) start the server in one terminal:   uvicorn fraud.serving.app:app --port 8000 --log-level warning
2) in a second terminal:               python scripts/benchmark_api.py --label pipeline
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud import config as C  # noqa: E402


def pct(a):
    a = np.asarray(a)
    return {"n": int(a.size), "mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "max": round(float(a.max()), 3)}


def load_payloads(n):
    df = pd.read_csv(C.DATA_SAMPLE / "unseen_transactions.csv", nrows=max(n, 1000))
    return [{"transaction_id": str(r[C.ID_COL]), **{c: float(r[c]) for c in C.RAW_FEATURES}}
            for r in df.to_dict("records")]


def sequential(url, payloads, n, warmup=100):
    client_ms, server_ms = [], []
    with httpx.Client(base_url=url, timeout=10) as c:
        for i in range(warmup):                                  # warm-up, not measured
            c.post("/predict", json=payloads[i % len(payloads)])
        for i in range(n):
            t0 = time.perf_counter()
            r = c.post("/predict", json=payloads[i % len(payloads)])
            client_ms.append((time.perf_counter() - t0) * 1000)  # full round trip
            r.raise_for_status()
            server_ms.append(r.json()["latency_ms"])             # time inside the handler
    return pct(client_ms), pct(server_ms)


async def concurrent(url, payloads, n, conc):
    lat, errors = [], 0
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(base_url=url, timeout=10, limits=httpx.Limits(max_connections=conc)) as c:
        async def one(i):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await c.post("/predict", json=payloads[i % len(payloads)])
                lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    return {**pct(lat), "concurrency": conc, "throughput_rps": round(n / wall, 1), "errors": errors}


def batch_endpoint(url, payloads, size, reps=50):
    lat = []
    body = {"transactions": payloads[:size]}
    with httpx.Client(base_url=url, timeout=30) as c:
        c.post("/predict/batch", json=body)
        for _ in range(reps):
            t0 = time.perf_counter()
            c.post("/predict/batch", json=body).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
    s = pct(lat)
    s.update(batch_size=size, records_per_s=round(size / (s["p50"] / 1000), 1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--label", default="pipeline")
    a = ap.parse_args()
    payloads = load_payloads(a.n)
    res = {"label": a.label}
    res["sequential_client"], res["sequential_server_handler"] = sequential(a.url, payloads, a.n)
    res["concurrent"] = [asyncio.run(concurrent(a.url, payloads, a.n, k)) for k in (4, 16)]
    res["batch_endpoint"] = [batch_endpoint(a.url, payloads, k) for k in (10, 100, 1000)]
    res["server_metrics"] = httpx.get(a.url + "/metrics").json()
    out = C.REPORTS / f"api_benchmark_{a.label}.json"
    out.write_text(json.dumps(res, indent=2))
    print("SEQUENTIAL client :", res["sequential_client"])
    print("SEQUENTIAL handler:", res["sequential_server_handler"])
    for r in res["concurrent"]:
        print(f"CONCURRENT x{r['concurrency']:<3}:", {k: r[k] for k in ['p50', 'p95', 'p99', 'throughput_rps', 'errors']})
    for r in res["batch_endpoint"]:
        print(f"BATCH {r['batch_size']:<5}     :", {k: r[k] for k in ['p50', 'p95', 'p99', 'records_per_s']})
    print("saved ->", out)


if __name__ == "__main__":
    main()