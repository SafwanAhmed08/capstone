import time
import json
import argparse
import sys
import paho.mqtt.client as mqtt
import ast

DEFAULT_BROKER = "broker.hivemq.com"
DEFAULT_PORT = 8000  # HiveMQ public broker websocket port
DEFAULT_TOPIC = "shrijit/test/topic"


def send_to_hivemq(payload, broker=DEFAULT_BROKER, port=DEFAULT_PORT, topic=DEFAULT_TOPIC, transport="websockets"):
    """Publish a JSON payload (string) to the MQTT broker/topic.

    payload: a JSON string (already json.dumps-ed) or bytes
    """
    # use websockets transport when connecting to port 8000
    client = mqtt.Client(transport=transport)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"🔌 Publisher connected to broker {broker}:{port}")
        else:
            print(f"⚠️ Publisher failed to connect, rc={rc}")

    client.on_connect = on_connect
    client.connect(broker, port, 60)
    client.loop_start()  # start background network loop

    # publish with QoS 1 to improve delivery chance
    result = client.publish(topic, payload, qos=1, retain=False)
    try:
        result.wait_for_publish()  # block until published
    except Exception:
        # some paho versions return a non-blocking object; ignore
        pass
    print(f"✅ Sent to topic '{topic}': {payload}")

    # give network loop a moment, then stop and disconnect
    time.sleep(0.2)
    client.loop_stop()
    client.disconnect()


def build_payload_from_args(attack: str = None, confidence: float = None, raw_json: str = None):
    if raw_json:
        # Some shells pass the JSON wrapped in extra quotes; strip common
        # surrounding single/double quotes before attempting to parse.
        s = raw_json.strip()
        if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ('"', "'")):
            s = s[1:-1]
        # First try strict JSON
        try:
            obj = json.loads(s)
            return json.dumps(obj, ensure_ascii=False)
        except Exception as json_exc:
            # Fallback: accept Python-style dict literals (single quotes) via ast.literal_eval
            try:
                obj = ast.literal_eval(s)
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                raise ValueError(f"Invalid JSON provided to --json: {json_exc}")

    if attack:
        obj = {"Attack": attack}
        if confidence is not None:
            try:
                obj["confidence"] = float(confidence)
            except Exception:
                obj["confidence"] = confidence
        return json.dumps(obj, ensure_ascii=False)

    # fallback: error
    raise ValueError("No valid payload specified (use --attack or --json)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple MQTT publisher for sending JSON messages to the detection pipeline")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT broker port (8000 for websockets)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="MQTT topic to publish to")
    parser.add_argument("--transport", default="websockets", choices=["websockets", "tcp"], help="MQTT transport to use")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--attack", help="Attack name to send (e.g. DDoS_TCP)")
    group.add_argument("--json", dest="raw_json", help='Raw JSON string to send (e.g. {"Attack":"DDoS_TCP","confidence":0.86})')
    parser.add_argument("--confidence", type=float, help="Optional confidence value (0..1) when using --attack")
    parser.add_argument("--file", help="Read JSON payload from a file (takes precedence over --attack/--json)")

    args = parser.parse_args(argv)

    # prepare payload
    payload = None
    try:
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                raw = f.read()
            # validate/normalize
            obj = json.loads(raw)
            payload = json.dumps(obj, ensure_ascii=False)
        else:
            payload = build_payload_from_args(attack=args.attack, confidence=args.confidence, raw_json=args.raw_json)
    except Exception as e:
        print(f"Error building payload: {e}")
        parser.print_help()
        sys.exit(2)

    send_to_hivemq(payload, broker=args.broker, port=args.port, topic=args.topic, transport=args.transport)


if __name__ == "__main__":
    main()
