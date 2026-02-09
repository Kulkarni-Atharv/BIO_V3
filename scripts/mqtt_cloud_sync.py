
import time
import json
import sys
import os
import mysql.connector
import paho.mqtt.client as mqtt
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MQTTSync")

# Get absolute path to BIO/ and add it to sys.path
# This assumes the script is run from the project root or scripts/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from shared.config import (
        MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT,
        MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
    )
except ImportError:
    # Fallback if running standalone without shared/config.py available or path issues
    # But for this project we assume standard structure
    logger.error("Could not import configuration. Make sure you are in the project environment.")
    sys.exit(1)

# Since this runs on the LAPTOP, MYSQL_HOST should likely be '127.0.0.1' or the LAN IP
# We will use the config values, but user might need to ensure they work locally.
# If config says '192.168.1.100' (Laptop IP) and this runs ON the laptop, it works fine.

class CloudSyncer:
    def __init__(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_connected = False
        
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            logger.info("Connected to MQTT Broker")
        else:
            logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        logger.warning("Disconnected from MQTT Broker")

    def connect_mqtt(self):
        try:
            logger.info(f"Connecting to MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start() # Start background thread for MQTT
        except Exception as e:
            logger.error(f"MQTT Connection Failed: {e}")

    def get_db_connection(self):
        try:
            return mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                port=MYSQL_PORT
            )
        except mysql.connector.Error as err:
            logger.error(f"Database Connection Error: {err}")
            return None

    def fetch_unsynced_records(self, conn):
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM attendance_log WHERE synced = 0 LIMIT 10")
            return cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(f"Error fetching records: {err}")
            return []

    def mark_synced(self, conn, record_idx):
        if not record_idx: return
        try:
            cursor = conn.cursor()
            placeholders = ', '.join(['%s'] * len(record_idx))
            sql = f"UPDATE attendance_log SET synced = 1 WHERE id IN ({placeholders})"
            cursor.execute(sql, record_idx)
            conn.commit()
            logger.info(f"Marked {len(record_idx)} records as synced.")
        except mysql.connector.Error as err:
            logger.error(f"Error updating records: {err}")

    def run(self):
        self.connect_mqtt()
        
        logger.info("Starting Sync Loop. Press Ctrl+C to stop.")
        while True:
            try:
                # 1. Connect to DB
                conn = self.get_db_connection()
                if not conn:
                    time.sleep(10)
                    continue

                # 2. Fetch Unsynced
                records = self.fetch_unsynced_records(conn)
                
                if records:
                    synced_ids = []
                    for record in records:
                        if self.mqtt_connected:
                            # Payload
                            payload = {
                                "id": record['id'],
                                "device_id": record['device_id'],
                                "name": record['name'],
                                "timestamp": record['timestamp'],
                                "status": "present" # Logic could be added here
                            }
                            msg = json.dumps(payload)
                            
                            # Publish
                            info = self.mqtt_client.publish(MQTT_TOPIC, msg, qos=1)
                            info.wait_for_publish() # Blocking to ensure sent
                            
                            if info.is_published():
                                logger.info(f"Published: {record['name']}")
                                synced_ids.append(record['id'])
                            else:
                                logger.error("Failed to publish message")
                        else:
                            logger.warning("MQTT not connected, waiting...")
                            break # Retry loop
                    
                    # 3. Mark as Synced
                    if synced_ids:
                        self.mark_synced(conn, synced_ids)
                
                conn.close()
                time.sleep(5) # Poll every 5 seconds

            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.mqtt_client.loop_stop()
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    syncer = CloudSyncer()
    syncer.run()
