import sys
import os
import json
import time
import logging
import ssl
import paho.mqtt.client as mqtt
from datetime import datetime, date, time as dt_time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, 
    MQTT_TOPIC_PREFIX, DEVICE_ID
)
from device.database import LocalDatabase

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MQTTSync")

class MQTTSyncService:
    def __init__(self):
        self.db = LocalDatabase()
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        # SSL Context for EMQX Cloud (Port 8883)
        if MQTT_PORT == 8883:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE # Adjust if using CA certs
            self.client.tls_set_context(context)

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish
        
        self.topic = f"{MQTT_TOPIC_PREFIX}/{DEVICE_ID}/updates"
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            self.connected = True
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from MQTT Broker (rc={rc})")
        self.connected = False

    def on_publish(self, client, userdata, mid):
        # logger.debug(f"Message {mid} published.")
        pass

    def json_serializer(self, obj):
        """Helper to serialize dates and times for JSON"""
        if isinstance(obj, (datetime, date, dt_time)):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def run(self):
        logger.info(f"Starting MQTT Sync Service...")
        logger.info(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
        logger.info(f"Topic: {self.topic}")

        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start() 
            
            # Allow time for connection
            time.sleep(2)

            while True:
                if self.connected:
                    self.sync_records()
                else:
                    logger.warning("Waiting for connection...")
                
                time.sleep(5) # Sync Interval

        except KeyboardInterrupt:
            logger.info("Stopping Sync Service...")
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Critical Error: {e}")

    def sync_records(self):
        # 1. Fetch Unsynced Records (Batch of 10)
        records = self.db.get_unsynced_records(limit=10)
        
        if not records:
            return

        logger.info(f"Found {len(records)} unsynced records.")
        
        synced_ids = []
        
        for record in records:
            try:
                # 2. Serialize
                payload = json.dumps(record, default=self.json_serializer)
                
                # 3. Publish
                info = self.client.publish(self.topic, payload, qos=1)
                info.wait_for_publish(timeout=2)
                
                if info.is_published():
                    synced_ids.append(record['id'])
                    logger.info(f"Published Record ID: {record['id']}")
                else:
                    logger.warning(f"Failed to publish Record ID: {record['id']}")
            
            except Exception as e:
                logger.error(f"Error publishing record {record.get('id')}: {e}")

        # 4. Mark Synced
        if synced_ids:
            self.db.mark_as_synced(synced_ids)
            logger.info(f"Marked {len(synced_ids)} records as synced.")

if __name__ == "__main__":
    service = MQTTSyncService()
    service.run()
