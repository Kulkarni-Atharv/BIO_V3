
import cv2
import platform
import threading
import time

from shared.config import CAMERA_INDEX


def open_usb_camera(index=CAMERA_INDEX, width=640, height=480):
    """Open a USB (UVC) camera. Uses V4L2 on Linux/Raspberry Pi, OpenCV default elsewhere."""
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        # MJPG lets most USB webcams deliver 640x480 @ 30fps over USB 2.0
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
    if not cap.isOpened():
        print(f"[Camera] Could not open USB camera at index {index}. "
              f"Run test_camera.py to find the right index and set CAMERA_INDEX in shared/config.py.")
    return cap


class Camera:
    def __init__(self, source=CAMERA_INDEX):
        self.source = source
        self.cap = open_usb_camera(self.source)
        self.ret = False
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        if not self.cap.isOpened():
            self.cap = open_usb_camera(self.source)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.01) # Small sleep to reduce CPU usage

    def get_frame(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
