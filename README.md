# Smart Attendance System (BIO)

## How to Run

This system consists of two parts: the **Server** (Central PC) and the **Device** (Client/Camera). You need to run them separately.

### Prerequisites
Make sure you have installed the dependencies:
```bash
./setup_env.sh
source venv/bin/activate
```

### 1. Run the Server
Open a terminal and run:
```bash
python3 server/main.py
```
*This starts the API on port 8000.*

### 2. Run the Device
Open a **new** terminal and run:
```bash
python3 device/main.py
```
*This opens the camera and starts marking attendance.*

### Directory Structure
- `device/`: Code for the camera and local database.
- `server/`: Code for the central API and SQL storage.
- `shared/`: Configuration files.
# BIO
