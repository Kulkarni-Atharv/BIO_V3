
import cv2
import threading
import time

class Camera:
    def __init__(self, source=0):
        self.source = source
        # Try GStreamer pipeline for Libcamera on Raspberry Pi
        gst_pipeline = (
            "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink"
        )
        # Attempt to open using GStreamer first
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
             print("[Camera] GStreamer pipeline failed, falling back to index 0...")
             self.cap = cv2.VideoCapture(self.source)
        self.ret = False
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        
        if not self.cap.isOpened():
            self.cap.open(self.source)
            
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
