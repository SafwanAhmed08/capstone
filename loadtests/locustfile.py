"""
Locust load test for a Rasa server REST webhook.

Usage (PowerShell):
  # Install once
  pip install locust

  # Start your Rasa server(s) in separate terminals, e.g.:
  # rasa run --enable-api --cors "*" --port 5005
  # rasa run actions --port 5055

  # Run Locust (web UI on http://localhost:8089)
  locust -f loadtests/locustfile.py --host http://localhost:5005
"""

import random
import uuid
from locust import HttpUser, task, between


class RasaUser(HttpUser):
    # Wait time between tasks per simulated user
    wait_time = between(0.5, 2.0)

    # A small pool of representative user messages. Customize to your domain.
    # messages = [
    #     "hello",
    #     "DDoS_TCP",
    #     "vulnerability scanning",
    #     "what is a MITM atack?",
    #     "list mitigations",
    #     "recommend defense",
    #     "status",
    # ]
    messages  = [
        "DDoS_TCP"
    ]

    @task(2)
    def talk_to_bot(self):
        msg = random.choice(self.messages)
        payload = {"sender": str(uuid.uuid4()), "message": msg}

        # Name helps aggregate results regardless of dynamic sender/message
        with self.client.post(
            "/webhooks/rest/webhook",
            json=payload,
            name="/webhooks/rest/webhook",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                _ = resp.json()
            except Exception:
                resp.failure("Invalid JSON in response")
