
import paho.mqtt.client as mqtt
import json

# Configuration
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "attendance/updates"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker: {BROKER}")
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
        print("Waiting for attendance messages...")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n[RECEIVED] Topic: {msg.topic}")
        print(f"Data: {json.dumps(payload, indent=2)}")
    except Exception as e:
        print(f"Error decoding message: {e}")

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nDisconnecting...")
    client.disconnect()
except Exception as e:
    print(f"Connection Error: {e}")
