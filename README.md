# Smart Attendance System (BIO_V3)

Face-recognition attendance terminal for Raspberry Pi / CM4 with a USB camera.
Faces are detected with **YuNet** and recognised with **MobileFaceNet** (both ONNX, run on CPU via OpenCV).
Attendance is stored locally in SQLite and synced to a LAN PC and/or an MQTT cloud dashboard when available.

## Features
- **Touch HMI** (PyQt5): live camera feed, attendance marking, user registration/deletion, employee list, settings screens.
- **USB camera** support (V4L2, 640x480 MJPG) - tested with Kreo Owl on `/dev/video0`.
- **Auto-training**: embeddings are rebuilt automatically when a user is added or removed.
- **Offline-first**: every punch is written to a local SQLite buffer first; nothing is lost without network.
- **Attendance logic**: IN/OUT punches, shift-based late / early-departure / overtime minutes (default shift 09:00-18:00, 15 min grace).
- **Sync**:
  - LAN - POSTs records to a FastAPI receiver on a PC in the same network.
  - Cloud - publishes records to an EMQX MQTT broker (TLS) and receives the employee list from the dashboard.

## Architecture
```text
 USB Camera --> hmi.py / device/main.py
                  |  YuNet (detect) -> align -> MobileFaceNet (embed) -> match
                  v
         data/attendance_buffer.db (SQLite)
            |                         |
   device/uploader.py          device/mqtt_sync.py
   (HTTP, LAN)                 (MQTT over TLS, internet)
            v                         v
   server/api.py on PC         EMQX broker <--> Dashboard
   data/server_attendance.db
```

## Directory Structure
```text
BIO_V3/
|-- hmi.py                 # Main touchscreen application (PyQt5)
|-- test_camera.py         # Lists working camera ports
|-- requirements.txt
|-- setup_env.sh
|-- assets/                # ONNX models (downloaded, not in git)
|-- data/                  # Faces, embeddings, SQLite DBs (created at runtime, not in git)
|-- core/
|   |-- alignment.py       # Face alignment
|   |-- face_encoder.py    # Builds embeddings from data/known_faces
|   `-- recognizer.py      # Detection + recognition
|-- device/
|   |-- camera.py          # USB camera helper (open_usb_camera, Camera)
|   |-- database.py        # Local SQLite: attendance_log, shifts, users
|   |-- main.py            # Headless mode (no HMI)
|   |-- uploader.py        # LAN sync to server/api.py
|   `-- mqtt_sync.py       # Cloud sync over MQTT
|-- server/
|   |-- api.py             # FastAPI LAN receiver
|   |-- database.py        # Server-side SQLite
|   `-- start_server.bat   # Windows launcher
|-- shared/
|   `-- config.py          # All settings
`-- scripts/               # Model download, dataset capture, diagnostics, tests
```

## Hardware
- Raspberry Pi 4 / CM4 (Raspberry Pi OS 64-bit recommended)
- USB webcam (UVC) - e.g. Kreo Owl
- Touch display for the HMI

## Installation (Raspberry Pi)
The project path below contains a space, so keep it in quotes.

```bash
# 1. System packages (PyQt5 from apt - pip cannot build it on the Pi)
sudo apt update
sudo apt install -y python3-full python3-venv python3-pip python3-pyqt5 libgl1 v4l-utils git
sudo usermod -aG video $USER          # log out / reboot after this

# 2. Get the code
cd "/home/autonex/Face Recognition"
git clone https://github.com/Kulkarni-Atharv/BIO_V3.git
cd BIO_V3

# 3. Virtual environment (system-site-packages so apt's PyQt5 is visible)
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install numpy requests paho-mqtt opencv-python-headless

# 4. Download models into assets/
python3 scripts/download_models.py

# 5. Verify
python3 -c "import cv2, numpy, paho.mqtt.client, PyQt5; print('OK', cv2.__version__)"
v4l2-ctl --list-devices
python3 test_camera.py
```

On the PC that runs the LAN receiver, install the server packages instead:
```bash
pip install fastapi uvicorn python-multipart pydantic requests
```

## Configuration
Edit [shared/config.py](shared/config.py):

| Setting | Purpose | Default |
|---|---|---|
| `DEVICE_ID` | ID of this terminal | `"1"` |
| `CAMERA_INDEX` | USB camera index (`/dev/videoN`) | `0` |
| `DETECTION_THRESHOLD` | YuNet face score threshold | `0.6` |
| `RECOGNITION_THRESHOLD` | Similarity needed to accept a match | `0.70` |
| `VERIFICATION_FRAMES` | Consecutive matching frames before marking | `5` |
| `LAN_SERVER_IP` / `LAN_SERVER_PORT` | PC running `server/api.py` | `192.168.1.100:8000` |
| `MQTT_BROKER` / `MQTT_PORT` / `MQTT_USERNAME` / `MQTT_PASSWORD` | EMQX cloud broker (TLS) | - |
| `MQTT_TOPIC_*` | MQTT topics (see below) | - |
| `RELAY_GPIO_PIN` | GPIO (BCM) driving the machine relay / LED | `17` |
| `RELAY_ACTIVE_HIGH` | `False` for relay boards that switch ON with a LOW signal | `False` |
| `RELAY_ON_SECONDS` | Relay ON time after ACCESS GRANTED (`0` = until hmi.py stops) | `0` |

The relay is handled by [device/relay.py](device/relay.py) (uses `gpiozero`, preinstalled on Raspberry Pi OS).
The relay is OFF until a face is authenticated, then stays ON until hmi.py stops. For active-low boards OFF releases
the pin instead of driving it HIGH (3.3V does not switch a 5V board off).
Test the wiring with `python3 device/relay.py` - OFF for 2 s, ON for 2 s, then OFF.

### Machine output: relay or Beckhoff controller
The machine enable signal can go to the GPIO relay **or** to a Beckhoff controller (CX7000, TwinCAT 3) over ADS:

```bash
python3 hmi.py                     # relay (default, MACHINE_OUTPUT in shared/config.py)
python3 hmi.py --output relay
python3 hmi.py --output beckhoff   # writes BECKHOFF_VARIABLE (MAIN.bCmd) TRUE / FALSE
```

| Setting | Purpose | Default |
|---|---|---|
| `MACHINE_OUTPUT` | Output used when `--output` is not given | `"relay"` |
| `BECKHOFF_AMS_NET_ID` | AMS Net ID of the controller | `5.177.87.98.1.1` |
| `BECKHOFF_PLC_IP` | Controller IP address | `169.254.99.87` |
| `BECKHOFF_VARIABLE` | BOOL variable: TRUE = machine ON | `MAIN.bCmd` |
| `BECKHOFF_TIMEOUT_MS` | ADS timeout | `1000` |

ACCESS GRANTED writes TRUE, STOP writes FALSE, closing hmi.py writes FALSE. If the controller cannot be reached the HMI
shows **MACHINE NOT STARTED** (on START) or **Could not stop machine** (on STOP, the STOP button stays available).
Beckhoff mode needs `pip install pyads` and an ADS route between the Pi and the controller.
Interactive test (on / off / status / quit): `python3 device/beckhoff.py`.

## Usage
Always activate the environment first:
```bash
cd "/home/autonex/Face Recognition/BIO_V3"
source venv/bin/activate
```

| What | Command | Runs on |
|---|---|---|
| Touch HMI (recommended) | `python3 hmi.py` | Pi |
| Headless recognition (no HMI) | `python3 device/main.py` | Pi |
| Cloud sync of attendance | `python3 device/mqtt_sync.py` | Pi |
| LAN receiver | `python -m uvicorn server.api:app --host 0.0.0.0 --port 8000` (or `server\start_server.bat` on Windows) | PC |
| Capture faces from terminal | `python3 scripts/capture_dataset.py` | Pi |

### Main screen (HMI) - machine start / stop
The HMI starts full screen (designed for the 5" 1280x720 touch panel): live camera on one half, controls on the other
(camera on top if the display is in portrait).
1. Press **START** to turn the machine ON - the face is scanned for up to 10 s.
2. The same person must match on `VERIFICATION_FRAMES` (default 3) frames; an unclear frame does not reset the count.
3. **ACCESS GRANTED** (name and ID) switches the relay ON and the round button becomes a red **STOP** button.
   **ACCESS DENIED / NO FACE DETECTED** keeps the machine OFF.
4. Press **STOP** to switch the relay OFF immediately (no face scan needed).

The footer shows **MACHINE ON / MACHINE OFF**. Recognition only runs during a scan, so the device is idle otherwise.
**MENU** opens settings and user management. With a keyboard attached, **F11** toggles full screen and **Esc** leaves it.

### Registering a user (HMI) - guided
1. Open **User Mgt -> Add User**, enter ID and name, press **Start Scanning**.
2. Follow the on-screen instructions. Samples are captured in three steps:

   | Step | Instruction | Samples |
   |---|---|---|
   | 1 | Look directly at the camera | 12 |
   | 2 | Turn your head slightly to your LEFT | 9 |
   | 3 | Turn your head slightly to your RIGHT | 9 |

   A frame is only saved when it passes every check; otherwise the screen tells the user what to fix:
   *No face detected*, *Only one person in front of the camera*, *Move your face closer / farther away*,
   *Keep your face in the centre of the circle*, *Too dark / Too bright*, *Hold still* (blur), and the pose instruction for the current step.
   The oval and face box turn **green** while capturing and **orange** when the user needs to adjust.
3. Training runs automatically. Registration is marked **complete only if at least 20 samples produced a usable face embedding**; otherwise the user is asked to scan again.
4. **Cancel** during scanning removes the samples captured in that session.
5. Tapping a text box (name, ID, settings) opens the on-screen keyboard; **DONE** closes it.

Thresholds (face size, centring, brightness, sharpness, head-turn angle, samples per step) are in [core/face_guide.py](core/face_guide.py).

Face images are stored in `data/known_faces/<id>_<name>/` (`front_*.jpg`, `left_*.jpg`, `right_*.jpg`); embeddings in `data/embeddings.npy` and `data/names.json`.

## MQTT Topics
| Topic | Direction | Content |
|---|---|---|
| `p/a/1/updates` | device -> broker | Attendance records |
| `p/a/1/request-users` | device -> broker | Request the employee list |
| `p/a/1/receive-users` | broker -> device | `{"users": [{"user_id": ..., "name": ...}, ...]}` |

## LAN API
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/attendance` | List of attendance records from a device |
| `GET` | `/api/attendance` | All received records |
| `GET` | `/health` | Health check |

## Local Database (`data/attendance_buffer.db`)
- `attendance_log` - each punch with IN/OUT, status, late/early/overtime minutes, confidence, and `lan_synced` / `mqtt_synced` flags.
- `shifts` - shift timings and grace rules (seeded with *General Shift*).
- `users` - employee master list received from the dashboard.

Inspect with `python3 scripts/inspect_db.py`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Could not open USB camera at index N` | Run `v4l2-ctl --list-devices` / `python3 test_camera.py` and set `CAMERA_INDEX`. Make sure the user is in the `video` group. |
| Black or frozen preview | Try another USB port; close other apps using the camera (`sudo fuser /dev/video0`). |
| `ModuleNotFoundError: PyQt5` | Install `python3-pyqt5` with apt and create the venv with `--system-site-packages`. |
| Model file not found | Run `python3 scripts/download_models.py` (needs internet). |
| LAN records not syncing | Check `LAN_SERVER_IP` and that port 8000 is allowed in the PC firewall. |

## Known Issues
- `server/main.py` and `server/mqtt_client.py` import `SERVER_PORT` / `MQTT_TOPIC`, which are not defined in `shared/config.py`. Start the server with the `uvicorn server.api:app` command above instead.
- `setup_env.sh` installs PyQt5 via pip, which fails on Raspberry Pi - follow the manual installation steps above.
