"""
Quick concurrent benchmark for Rasa NLU parse endpoint using asyncio/aiohttp.

Usage (PowerShell):
  pip install aiohttp
  python .\loadtests\quick_parse_benchmark.py --url http://localhost:5005/model/parse --requests 200 --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List

import aiohttp


SAMPLE_MESSAGES = [
    "hello",
    "help",
    "show mitigations for ddos",
    "mitigations for port scan",
    "what is mitm attack",
    "list mitigations",
]


async def worker(session: aiohttp.ClientSession, url: str, text: str, results: list[float], errors: list[str]):
    t0 = time.perf_counter()
    try:
        async with session.post(url, json={"text": text}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                errors.append(f"HTTP {resp.status}")
            else:
                await resp.json()
    except Exception as e:
        errors.append(str(e))
    finally:
        t1 = time.perf_counter()
        results.append((t1 - t0) * 1000.0)  # ms


async def run(url: str, total: int, concurrency: int) -> None:
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=15)
    results: List[float] = []
    errors: List[str] = []

    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bound_worker(i: int):
            async with sem:
                msg = SAMPLE_MESSAGES[i % len(SAMPLE_MESSAGES)]
                await worker(session, url, msg, results, errors)

        tasks = [asyncio.create_task(bound_worker(i)) for i in range(total)]
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        t1 = time.perf_counter()

    duration_s = t1 - t0
    ok = total - len(errors)
    rps = ok / duration_s if duration_s > 0 else 0

    p50 = statistics.median(results) if results else 0
    p95 = statistics.quantiles(results, n=100)[94] if len(results) >= 100 else (sorted(results)[int(0.95 * len(results)) - 1] if results else 0)
    p99 = statistics.quantiles(results, n=100)[98] if len(results) >= 100 else (sorted(results)[max(0, int(0.99 * len(results)) - 1)] if results else 0)

    print("--- Quick Parse Benchmark ---")
    print(f"URL           : {url}")
    print(f"Requests      : {total}")
    print(f"Concurrency   : {concurrency}")
    print(f"Duration (s)  : {duration_s:.2f}")
    print(f"Success       : {ok}")
    print(f"Errors        : {len(errors)}")
    if errors:
        print(f"First error   : {errors[0]}")
    print(f"Throughput RPS: {rps:.2f}")
    if results:
        print(f"Latency ms    : p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5005/model/parse", help="Rasa parse endpoint")
    ap.add_argument("--requests", type=int, default=100, help="Total number of requests")
    ap.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    args = ap.parse_args()

    asyncio.run(run(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
