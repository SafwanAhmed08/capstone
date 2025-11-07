#!/usr/bin/env python3
"""Helper to notify a running Rasa server about a detected threat via HTTP POST.

This avoids any Rasa SDK imports in the inference environment so you won't have
package/version conflicts. It posts an announcement and then triggers the
mitigation lookup by sending a follow-up message ("mitigation for <THREAT>").

Defaults match your UI (`conversational ai/ui/app.js`) which posts to
http://localhost:5006/webhooks/rest/webhook with sender 'user'.

Usage:
  python tools\notify_rasa.py --threat DDoS_TCP --confidence 0.98
  python tools\notify_rasa.py --sender user --threat DDoS_TCP --pcap tcp.pcap
"""
from __future__ import annotations

import time
import json
import argparse
from typing import Optional

import requests

DEFAULT_RASA = "http://localhost:5006/webhooks/rest/webhook"


def rasa_base_from_webhook(webhook_url: str) -> str:
    """Return the Rasa server base URL from a webhook URL.

    If webhook_url is the full webhook path, remove the '/webhooks/...' suffix.
    """
    # If the URL contains '/webhooks', strip from there
    idx = webhook_url.find('/webhooks')
    if idx != -1:
        return webhook_url[:idx]
    # otherwise assume provided URL is a base
    return webhook_url


def post_bot_event(rasa_base: str, sender: str, text: str, timeout: int = 5, external: bool = True):
    """Post a bot event to the conversation tracker so the UI can surface it.

    Adds metadata.source = 'external' so the UI can differentiate externally injected
    notifications from normal REST webhook replies and avoid duplication.
    """
    url = f"{rasa_base}/conversations/{sender}/tracker/events"
    payload = {"event": "bot", "text": text}
    if external:
        payload["metadata"] = {"source": "external"}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"Failed to post bot event to tracker: {e}")
        return None


def safe_post(url: str, payload: dict, timeout: int = 15, retries: int = 3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return resp.text
        except Exception as e:
            last_exc = e
            print(f"[attempt {attempt}] Error posting to Rasa: {e}")
            time.sleep(min(2 ** attempt, 8))
    raise last_exc


def notify_and_request_mitigation(
    threat_name: str,
    sender: str = "user",
    rasa_url: str = DEFAULT_RASA,
    confidence: Optional[float] = None,
    pcap: Optional[str] = None,
    announce_only: bool = False,
):
    announcement = f"🚨🚨THREAT DETECTED: {threat_name}"
    # if confidence is not None:
    #     try:
    #         announcement += f" | confidence={confidence:.2%}"
    #     except Exception:
    #         announcement += f" | confidence={confidence}"
    # if pcap:
    #     announcement += f" | pcap={pcap}"

    # Instead of posting the announcement as a user message (which may
    # trigger the LLM fallback), inject it directly as a bot event so the UI
    # displays the announcement without invoking NLU/policies.
    print(f"Injecting announcement into tracker as bot event for sender '{sender}'")
    try:
        rasa_base = rasa_base_from_webhook(rasa_url)
        post_bot_event(rasa_base, sender, announcement, timeout=5, external=True)
        ann_resp = {'injected_as': 'bot_event', 'text': announcement}
        print("Announcement injected as bot event:")
        print(json.dumps(ann_resp, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Warning: failed to post bot tracker event: {e}")
        # Fallback to posting as a user message if tracker injection fails
        print(f"Posting announcement as user message instead (fallback) to {rasa_url}")
        ann_payload = {"sender": sender, "message": announcement}
        ann_resp = safe_post(rasa_url, ann_payload)
        print("Announcement response:")
        print(json.dumps(ann_resp, indent=2, ensure_ascii=False))

    if announce_only:
        return ann_resp

    time.sleep(0.2)

    mitigation_text = f"{threat_name}"
    print(f"Posting mitigation trigger to Rasa: '{mitigation_text}'")
    mit_payload = {"sender": sender, "message": mitigation_text}
    mit_resp = safe_post(rasa_url, mit_payload)
    print("Mitigation response:")
    print(json.dumps(mit_resp, indent=2, ensure_ascii=False))
    return mit_resp


def main():
    parser = argparse.ArgumentParser(description="Notify a running Rasa server about a detected threat.")
    parser.add_argument("--sender", default="user", help="sender id to use for the conversation (default: 'user')")
    parser.add_argument("--rasa-url", default=DEFAULT_RASA, help="Rasa REST webhook URL")
    parser.add_argument("--threat", required=True, help="Threat name (e.g., DDoS_TCP)")
    parser.add_argument("--confidence", type=float, default=None, help="Optional confidence as a float (0..1)")
    parser.add_argument("--pcap", default=None, help="Optional pcap file name/path")
    parser.add_argument("--announce-only", action="store_true", help="Only send the announcement (no mitigation trigger)")

    args = parser.parse_args()

    try:
        notify_and_request_mitigation(
            threat_name=args.threat,
            sender=args.sender,
            rasa_url=args.rasa_url,
            confidence=args.confidence,
            pcap=args.pcap,
            announce_only=args.announce_only,
        )
    except Exception as e:
        print(f"Failed to notify Rasa: {e}")


if __name__ == "__main__":
    main()
