# Load and stress testing for Rasa

This folder contains a couple of simple load tests you can run locally against your Rasa server.

## Prerequisites

- Python 3.9+
- Rasa server running locally with API enabled (separate terminals):

```powershell
# Terminal 1 - Rasa server
rasa run --enable-api --cors "*" --port 5005

# Terminal 2 - Action server (if you use actions)
rasa run actions --port 5055
```

## Option A: Locust (web UI)

Locust simulates concurrent users sending messages to the REST webhook.

```powershell
pip install locust

# Start Locust (web UI at http://localhost:8089)
locust -f loadtests/locustfile.py --host http://localhost:5005
```

In the web UI, set:
- Number of users (e.g., 25)
- Spawn rate (e.g., 2 users/sec)

You’ll see requests, failures, and latency percentiles per endpoint.

## Option B: Quick parse benchmark (CLI)

Sends concurrent requests to the NLU parse endpoint and prints simple stats.

```powershell
pip install aiohttp
python .\loadtests\quick_parse_benchmark.py --url http://localhost:5005/model/parse --requests 200 --concurrency 20
```

Example output:
```
--- Quick Parse Benchmark ---
URL           : http://localhost:5005/model/parse
Requests      : 200
Concurrency   : 20
Duration (s)  : 6.42
Success       : 200
Errors        : 0
Throughput RPS: 31.15
Latency ms    : p50=60.2  p95=140.9  p99=220.4
```

## Tips

- Keep action server logic fast and avoid blocking I/O in the hot path.
- For realistic results, include a mix of short and longer user messages.
- Track P95/P99 latencies and error rate; consider setting an SLA (e.g., P95 < 700ms for full turn).
- For soak tests, let Locust run at a steady user count for 30–60 minutes and observe CPU/memory.