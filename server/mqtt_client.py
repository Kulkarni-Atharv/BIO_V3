
import paho.mqtt.client as mqtt
import json
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

logger = logging.getLogger("MQTTPublisher")

class MQTTPublisher:
    def __init__(self):
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.connected = False
        
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT Broker: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT Broker")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

    def publish_attendance(self, device_id, name, timestamp):
        if not self.connected:
            logger.warning("MQTT not connected, skipping publish")
            return

        from datetime import datetime
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        
        payload = {
            "device_id": device_id,
            "name": name,
            "timestamp": formatted_time,
            "status": "present"
        }
        
        try:
            self.client.publish(MQTT_TOPIC, json.dumps(payload))
            logger.info(f"Published to MQTT: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
