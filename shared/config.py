
import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Server Configuration
SERVER_IP = "127.0.0.1"  # Default to localhost for testing
SERVER_PORT = 8000
API_BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/api"
SERVER_DB_PATH = os.path.join(DATA_DIR, "server_attendance.db")

# Device Configuration
DEVICE_ID = "cm4_device_001"

# MySQL Configuration
MYSQL_HOST = "127.0.0.1" # REPLACE WITH YOUR LAPTOP IP
MYSQL_USER = "root"
MYSQL_PASSWORD = "Atharv@123" # REPLACE WITH YOUR PASSWORD
MYSQL_DB = "bio_attendance"
MYSQL_PORT = 3306

DB_PATH = os.path.join(DATA_DIR, "attendance_buffer.db")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")

# Models & Embeddings
YUNET_PATH = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
MOBILEFACENET_PATH = os.path.join(ASSETS_DIR, "MobileFaceNet.onnx")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
NAMES_FILE = os.path.join(DATA_DIR, "names.json")

# MQTT Configuration
# Using EMQX Public Broker for testing
MQTT_BROKER = "broker.emqx.io" 
MQTT_PORT = 1883
MQTT_TOPIC = "attendance/updates"

# Camera Configuration
CAMERA_INDEX = 0 
DETECTION_THRESHOLD = 0.6 
RECOGNITION_THRESHOLD = 0.65
VERIFICATION_FRAMES = 5 
