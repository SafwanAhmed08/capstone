# publish_sample.py (run on the other PC or same machine)
import json
import time
import paho.mqtt.client as mqtt

payload = {
  "source":"IDS-Ensemble",
  "timestamp":"2025-11-11T06:49:34.748787Z",
  "pcap":"tcp.pcap",
  "alerts":[
    {
      "ensemble_labels":["ddos_tcp"],
      "ensemble_probs":{"ddos_tcp":0.952}
    }
  ]
}


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to broker, rc={rc}")


def on_publish(client, userdata, mid):
    print(f"Message published (mid={mid})")
    userdata['published'] = True


def publish_payload(broker='broker.hivemq.com', port=8000, topic='shrijit/test/topic', transport='websockets'):
    userdata = {'published': False}
    client = mqtt.Client(transport=transport, userdata=userdata)
    client.on_connect = on_connect
    client.on_publish = on_publish

    # Use the network loop in a background thread so connect/publish callbacks run
    client.connect(broker, port, 60)
    client.loop_start()

    # publish and wait for on_publish to set the flag (with timeout)
    (rc, mid) = client.publish(topic, json.dumps(payload))
    print(f"publish returned rc={rc}, mid={mid}")

    timeout = 5.0
    start = time.time()
    while not userdata['published'] and (time.time() - start) < timeout:
        time.sleep(0.1)

    client.loop_stop()
    client.disconnect()
    if userdata['published']:
        print("Publish completed and client disconnected")
    else:
        print("Publish may not have completed before timeout")


if __name__ == '__main__':
    publish_payload()