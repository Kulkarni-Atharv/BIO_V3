"""
device/mqtt_sync.py
────────────────────
MQTT service with request-response pattern for employee sync:

  PUBLISH attendance  → p/a/1/updates
  PUBLISH request     → p/a/1/request-users   (ask dashboard to send employee list)
  SUBSCRIBE response  → p/a/1/receive-users   (receive [{user_id, name}, ...])

On connect: immediately sends a request-users message so the device
always gets a fresh employee list after coming online.
On every re-connect (internet restored): requests again to catch any
new employees added on the dashboard while offline.
"""

import sys, os, json, time, socket, logging, ssl
import paho.mqtt.client as mqtt
from datetime import datetime, date, time as dt_time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_TOPIC_PREFIX, MQTT_TOPIC_REQUEST_USERS, MQTT_TOPIC_RECEIVE_USERS, DEVICE_ID
)
from device.database import LocalDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MQTT_Sync")

SYNC_INTERVAL  = 10   # seconds between attendance publish cycles
RETRY_INTERVAL = 15   # seconds to wait when no internet


# ─── Internet probe ───────────────────────────────────────────────────────────

def _has_internet(host="8.8.8.8", port=53, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ─── Service ─────────────────────────────────────────────────────────────────

class MQTTSyncService:
    def __init__(self):
        self.db        = LocalDatabase()
        self.client    = mqtt.Client()
        self.connected = False

        # Derived topics
        self.pub_attendance   = f"{MQTT_TOPIC_PREFIX}/updates"
        self.pub_req_users    = MQTT_TOPIC_REQUEST_USERS   # publish to request list
        self.sub_recv_users   = MQTT_TOPIC_RECEIVE_USERS   # subscribe to receive list

        # Auth + TLS
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        if MQTT_PORT == 8883:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            self.client.tls_set_context(ctx)

        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message    = self._on_message

    # ── MQTT callbacks ────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("Connected to EMQX broker.")

            # Subscribe to receive employee list from dashboard
            client.subscribe(self.sub_recv_users, qos=1)
            logger.info("Subscribed: %s", self.sub_recv_users)

            # Immediately request the employee list so we are always up-to-date
            self._request_users()
        else:
            logger.error("MQTT connect failed (rc=%d).", rc)

    def _on_disconnect(self, client, userdata, rc):
        logger.warning("Disconnected from MQTT broker (rc=%d).", rc)
        self.connected = False

    def _on_message(self, client, userdata, msg):
        """Handle incoming messages."""
        try:
            raw   = msg.payload.decode("utf-8")
            topic = msg.topic

            print(f"\n[{topic}]\n{raw}\n")

            payload = json.loads(raw)

            if topic == self.sub_recv_users:
                # Dashboard may send: [...] (raw array) or {"users": [...]} (wrapped)
                if isinstance(payload, dict) and "users" in payload:
                    payload = payload["users"]   # unwrap
                elif isinstance(payload, dict):
                    payload = [payload]           # single user dict

                if isinstance(payload, list):
                    self.db.upsert_users(payload)
                    logger.info("Saved %d employees to local DB.", len(payload))
                else:
                    logger.warning("Unexpected payload type: %s", type(payload))
            else:
                logger.debug("Unhandled topic: %s", topic)

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s  |  raw=%s", e, msg.payload)
        except Exception as e:
            logger.error("Error in on_message: %s", e)

    # ── Request user list from dashboard ──────────────────────────────────────

    def _request_users(self):
        """Publish a request so the dashboard sends the employee list."""
        payload = json.dumps({"device_id": DEVICE_ID, "action": "get-users"})
        self.client.publish(self.pub_req_users, payload, qos=1)
        logger.info("Sent employee list request on: %s", self.pub_req_users)

    # ── JSON serialiser ───────────────────────────────────────────────────────

    @staticmethod
    def _serialise(obj):
        if isinstance(obj, (datetime, date, dt_time)):
            return str(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info("MQTT Sync Service starting...")
        logger.info("Attendance publish : %s", self.pub_attendance)
        logger.info("Request users      : %s", self.pub_req_users)
        logger.info("Receive users      : %s", self.sub_recv_users)

        try:
            while True:
                if not _has_internet():
                    logger.warning("No internet — retrying in %ds.", RETRY_INTERVAL)
                    self.connected = False
                    time.sleep(RETRY_INTERVAL)
                    continue

                if not self.connected:
                    logger.info("Internet detected — connecting to broker...")
                    try:
                        self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                        self.client.loop_start()
                        time.sleep(2)  # allow handshake + subscribe
                    except Exception as e:
                        logger.error("MQTT connect error: %s — retry in %ds.", e, RETRY_INTERVAL)
                        time.sleep(RETRY_INTERVAL)
                        continue

                if self.connected:
                    self._publish_attendance()

                time.sleep(SYNC_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopping MQTT Sync Service...")
            self.client.loop_stop()
            self.client.disconnect()

    # ── Publish pending attendance records ─────────────────────────────────────

    def _publish_attendance(self):
        records = self.db.get_unsynced_mqtt_records(limit=10)
        if not records:
            return

        logger.info("Publishing %d attendance records...", len(records))
        synced_ids = []

        for rec in records:
            payload = {
                "id":                      rec.get("id"),
                "user_id":                 rec.get("user_id"),
                "name":                    rec.get("name"),
                "device_id":               rec.get("device_id"),
                "punch_time":              rec.get("punch_time"),
                "punch_type":              rec.get("punch_type"),
                "attendance_status":       rec.get("attendance_status"),
                "late_minutes":            rec.get("late_minutes"),
                "early_departure_minutes": rec.get("early_departure_minutes"),
                "overtime_minutes":        rec.get("overtime_minutes"),
                "confidence":              rec.get("confidence"),
            }
            try:
                msg  = json.dumps(payload, default=self._serialise)
                info = self.client.publish(self.pub_attendance, msg, qos=1)
                info.wait_for_publish(timeout=3)
                if info.is_published():
                    synced_ids.append(rec["id"])
                else:
                    logger.warning("Publish timeout for record %s.", rec["id"])
            except Exception as e:
                logger.error("Error publishing record %s: %s", rec.get("id"), e)

        if synced_ids:
            self.db.mark_mqtt_synced(synced_ids)
            logger.info("MQTT synced: %d records marked.", len(synced_ids))


if __name__ == "__main__":
    MQTTSyncService().run()
