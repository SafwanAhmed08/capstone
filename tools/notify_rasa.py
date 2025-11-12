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
try:
    import paho.mqtt.client as mqtt
except Exception:  # keep script usable even if paho-mqtt isn't installed
    mqtt = None
from datetime import datetime
import os

# Where to store incoming alerts for later retrieval by Rasa actions
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'conversational ai', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'network_alerts.jsonl')


def ensure_log_dir():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        pass


def append_alert_to_log(raw_data: dict, parsed_attack: str = None, confidence: float = None, pcap: str = None):
    ensure_log_dir()
    entry = {
        'received_at': datetime.utcnow().isoformat() + 'Z',
        'parsed_attack': parsed_attack,
        'confidence': confidence,
        'pcap': pcap,
        'raw': raw_data,
    }
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Failed to write alert to log: {e}")

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
    # threat is required for one-shot CLI mode, but not when --mqtt-listen is used
    parser.add_argument("--threat", required=False, help="Threat name (e.g., DDoS_TCP)")
    parser.add_argument("--confidence", type=float, default=None, help="Optional confidence as a float (0..1)")
    parser.add_argument("--pcap", default=None, help="Optional pcap file name/path")
    parser.add_argument("--announce-only", action="store_true", help="Only send the announcement (no mitigation trigger)")
    # MQTT listener options
    parser.add_argument("--mqtt-listen", action="store_true", help="Start an MQTT listener and forward incoming messages to Rasa")
    parser.add_argument("--mqtt-broker", default="broker.hivemq.com", help="MQTT broker hostname")
    parser.add_argument("--mqtt-port", type=int, default=8000, help="MQTT broker port (default 8000 for websockets)")
    parser.add_argument("--mqtt-topic", default="shrijit/test/topic", help="MQTT topic to subscribe to")
    parser.add_argument("--mqtt-transport", default="websockets", choices=["websockets", "tcp"], help="MQTT transport to use (websockets or tcp)")

    args = parser.parse_args()

    # If not running as an MQTT listener, --threat must be provided
    if not args.mqtt_listen and not args.threat:
        parser.error("the following arguments are required: --threat (unless --mqtt-listen is used)")

    if args.mqtt_listen:
        # start MQTT listener (blocking)
        if mqtt is None:
            print("paho-mqtt is not installed. Install with: pip install paho-mqtt")
            return

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"🔌 MQTT connected to {args.mqtt_broker}:{args.mqtt_port}")
                client.subscribe(args.mqtt_topic, qos=1)
                print(f"� Subscribed to topic '{args.mqtt_topic}'")
            else:
                print(f"⚠️ MQTT failed to connect, rc={rc}")

        def on_message(client, userdata, msg):
            try:
                payload = msg.payload.decode(errors='ignore')
            except Exception:
                payload = msg.payload
            print(f"📥 MQTT message on {msg.topic}: {payload}")

            # Parse JSON payload and support both old simple format and the
            # new ensemble-style payload described by the user.
            try:
                data = json.loads(payload)
            except Exception:
                print("⚠️ MQTT payload is not valid JSON; ignoring")
                return

            attack = None
            confidence = None

            # New ensemble-style message: prefer alerts[0].ensemble_labels[0]
            alerts = data.get("alerts") or data.get("Alerts")
            if isinstance(alerts, list) and len(alerts) > 0:
                first = alerts[0]
                ensemble_labels = first.get("ensemble_labels")
                if isinstance(ensemble_labels, list) and ensemble_labels:
                    attack = ensemble_labels[0]

                # Get confidence from ensemble_probs if possible
                probs = first.get("ensemble_probs")
                if attack and isinstance(probs, dict):
                    # try exact label, then lowercased key
                    confidence = probs.get(attack)
                    if confidence is None:
                        confidence = probs.get(attack.lower())
                    try:
                        if confidence is not None:
                            confidence = float(confidence)
                    except Exception:
                        confidence = None

                # fallback to binary_prob
                if confidence is None and first.get("binary_prob") is not None:
                    try:
                        confidence = float(first.get("binary_prob"))
                    except Exception:
                        confidence = None

            # Backwards-compatible keys if not using the new format
            if not attack:
                attack = data.get("Attack") or data.get("attack") or data.get("threat")
            if confidence is None:
                confidence = data.get("confidence") or data.get("Confidence")
                try:
                    if confidence is not None:
                        confidence = float(confidence)
                except Exception:
                    pass

            if not attack:
                print("⚠️ MQTT message missing attack information; ignoring")
                return

            try:
                # Persist incoming alert (raw + parsed) for later "recent logs" queries
                try:
                    append_alert_to_log(raw_data=data, parsed_attack=attack, confidence=confidence, pcap=data.get('pcap') or args.pcap)
                except Exception as e:
                    print(f"Warning: failed to append alert to log: {e}")

                mit_resp = notify_and_request_mitigation(
                    threat_name=attack,
                    sender=args.sender,
                    rasa_url=args.rasa_url,
                    confidence=confidence,
                    pcap=args.pcap,
                    announce_only=args.announce_only,
                )
                # print a compact summary of the mitigation response when possible
                if mit_resp is not None:
                    try:
                        if isinstance(mit_resp, dict) or isinstance(mit_resp, list):
                            print("Mitigation response (parsed JSON):")
                            print(json.dumps(mit_resp, indent=2, ensure_ascii=False))
                        else:
                            print("Mitigation response:")
                            print(mit_resp)
                    except Exception:
                        print(mit_resp)
            except Exception as e:
                print(f"Error while forwarding to Rasa: {e}")

        print(f"👂 Starting MQTT listener -> forwarding to Rasa at {args.rasa_url}")
        client = mqtt.Client(transport=args.mqtt_transport)
        client.on_connect = on_connect
        client.on_message = on_message
        try:
            client.connect(args.mqtt_broker, args.mqtt_port, 60)
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return
        client.loop_forever()

    else:
        try:
            # Log CLI-invoked alert as well
            try:
                append_alert_to_log(raw_data={"threat": args.threat, "pcap": args.pcap}, parsed_attack=args.threat, confidence=args.confidence, pcap=args.pcap)
            except Exception:
                pass
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
