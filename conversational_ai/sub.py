import paho.mqtt.client as mqtt
import time

BROKER = "broker.hivemq.com"
PORT = 8000  # websocket port on public HiveMQ broker
TOPIC = "shrijit/test/topic"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("🔌 Subscriber connected to broker")
        client.subscribe(TOPIC, qos=1)
        print(f"� Subscribed to topic '{TOPIC}'")
    else:
        print(f"⚠️ Subscriber failed to connect, rc={rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
    except Exception:
        payload = msg.payload
    print(f"📥 Received message: {payload} from topic: {msg.topic}")

def retrieve_from_hivemq():
    # use websockets transport for port 8000
    client = mqtt.Client(transport="websockets")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    print(f"👂 Connecting and listening on topic '{TOPIC}'...")
    # loop_forever will handle reconnects and keep the process alive
    client.loop_forever()


if __name__ == "__main__":
    retrieve_from_hivemq()
