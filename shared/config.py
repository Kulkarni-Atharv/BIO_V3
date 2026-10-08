
import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── Device Configuration ─────────────────────────────────────────────────────
DEVICE_ID = "1"

# ─── Local SQLite Databases ───────────────────────────────────────────────────
# On-device DB (CM4 / Raspberry Pi) — no MySQL needed
DB_PATH = os.path.join(DATA_DIR, "attendance_buffer.db")

# Server-side DB (created automatically on the LAN receiver PC)
SERVER_DB_PATH = os.path.join(DATA_DIR, "server_attendance.db")

# ─── Face Data ────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
NAMES_FILE      = os.path.join(DATA_DIR, "names.json")

# ─── Models ───────────────────────────────────────────────────────────────────
YUNET_PATH        = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
MOBILEFACENET_PATH = os.path.join(ASSETS_DIR, "MobileFaceNet.onnx")

# ─── LAN Sync (PC / Laptop on same network) ───────────────────────────────────
# Set this to the IP address of the laptop/PC running server/api.py
LAN_SERVER_IP   = "192.168.1.100"   # <-- CHANGE to your PC's local IP
LAN_SERVER_PORT = 8000
API_BASE_URL    = f"http://{LAN_SERVER_IP}:{LAN_SERVER_PORT}/api"

# ─── MQTT Cloud Sync (EMQX — only used when internet is available) ─────────────
MQTT_BROKER       = "m318c100.ala.us-east-1.emqxsl.com"
MQTT_PORT         = 8883  # SSL
MQTT_USERNAME     = "autonex"
MQTT_PASSWORD     = "Autonex@2025"
MQTT_TOPIC_PREFIX          = "p/a/1"               # Publish attendance: p/a/1/updates
MQTT_TOPIC_REQUEST_USERS   = "p/a/1/request-users"  # Publish — ask dashboard for employee list
MQTT_TOPIC_RECEIVE_USERS   = "p/a/1/receive-users"  # Subscribe — receive employee list from dashboard

# ─── Camera & Recognition ─────────────────────────────────────────────────────
CAMERA_INDEX          = 0
DETECTION_THRESHOLD   = 0.6
RECOGNITION_THRESHOLD = 0.70
VERIFICATION_FRAMES   = 3   # same person must match this many frames during a scan

# ─── Machine Relay (LED for now) ─────────────────────────────────────────────
RELAY_GPIO_PIN    = 17     # BCM numbering (physical pin 11)
RELAY_ACTIVE_HIGH = False  # relay board switches ON with a LOW signal (active-low); True for active-high boards
RELAY_ON_SECONDS  = 0      # 0 = stay ON after ACCESS GRANTED until hmi.py stops; >0 = ON for that many seconds
