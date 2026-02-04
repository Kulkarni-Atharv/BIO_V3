
import os

# Server Configuration
SERVER_IP = "127.0.0.1"  # Default to localhost for testing
SERVER_PORT = 8000
API_BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/api"

# Device Configuration
DEVICE_ID = "cm4_device_001"
DB_PATH = "attendance_buffer.db"
KNOWN_FACES_DIR = "known_faces"

# MQTT Configuration
# Using EMQX Public Broker for testing
MQTT_BROKER = "broker.emqx.io" 
MQTT_PORT = 1883
MQTT_TOPIC = "attendance/updates"

# Camera Configuration
CAMERA_INDEX = 0 
