# Smart Attendance System (BIO_V2)
A Python-based Face Recognition Attendance System using **MobileFaceNet** for efficient inference on CPU/Edge devices.

## Features
-   **Offline HMI**: Native PyQt5 interface for live monitoring and user management.
-   **Auto-Training**: Automatically updates the model when users are added or removed.
-   **Edge Compatible**: Optimized for Raspberry Pi (Raspberry Pi OS).
-   **Dual Mode**: Supports both Headless (Service) and GUI (HMI) modes.

## Directory Structure
```text
BIO/
├── assets/                  # ONNX Models (YuNet, MobileFaceNet)
├── core/                    # Shared core logic (Encoder, Recognizer)
├── data/                    # Dynamic data (Faces, Embeddings, Database)
├── device/                  # Device-specific code (Camera, Offline DB Sync)
├── scripts/                 # Utility scripts (Download models, Capture data)
├── server/                  # Optional Backend API
└── hmi.py                   # Main Graphical Interface
```

## Setup
1.  **Install Dependencies:**
    ```bash
    ./setup_env.sh
    source venv/bin/activate
    ```
2.  **Download Models:**
    ```bash
    python3 scripts/download_models.py
    ```

## Usage

### Option 1: HMI (Recommended)
Run the graphical interface to manage users and view live feed:
```bash
./venv/bin/python hmi.py
```
-   **Add User**: Click "Add New User", enter details, capture, and wait for auto-train.
-   **Delete User**: Click "Delete User", select from list.

### Option 2: Headless Device
Run as a background service (monitor only):
```bash
./venv/bin/python device/main.py
```

### Option 3: Backend Server (Optional)
Run the central server for logs aggregation:
```bash
python3 server/main.py
```
