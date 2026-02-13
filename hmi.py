
import sys
import cv2
import os
import time
import numpy as np
from datetime import datetime

# Try to import picamera2 for Raspberry Pi CSI cameras
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QMessageBox, QFrame, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush

# Import modules
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from core.face_encoder import FaceEncoder
from shared.config import DEVICE_ID, KNOWN_FACES_DIR, VERIFICATION_FRAMES

# --- STYLESHEETS ---
STYLE_MAIN = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
}
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
QLineEdit:focus {
    border: 2px solid #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 15px;
    padding: 15px;
    font-size: 18px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QListWidget {
    background-color: #313244;
    border-radius: 10px;
    padding: 10px;
    color: #cdd6f4;
    font-size: 14px;
}
"""

# --- CUSTOM WIDGETS ---
class OverlayLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            background-color: rgba(0, 0, 0, 150);
            color: #a6e3a1;
            font-size: 24px;
            font-weight: bold;
            border-radius: 20px;
        """)
        self.hide()

    def show_message(self, text, duration=2000):
        self.setText(text)
        self.show()
        QTimer.singleShot(duration, self.hide)

class CircularProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setFixedSize(200, 200)

    def set_value(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        painter.translate(rect.center())
        
        # Background Circle
        pen = QPen(QColor("#45475a"), 10)
        painter.setPen(pen)
        painter.drawEllipse(-80, -80, 160, 160)
        
        # Progress Arc
        if self.value > 0:
            pen.setColor(QColor("#89b4fa"))
            painter.setPen(pen)
            span = int(-self.value * 3.6 * 16) # 360 degrees
            painter.drawArc(-80, -80, 160, 160, 90 * 16, span)

        # Text
        painter.setPen(QColor("#cdd6f4"))
        painter.setFont(QFont("Segoe UI", 24, QFont.Bold))
        text = f"{int(self.value)}%"
        fm = painter.fontMetrics()
        w = fm.width(text)
        h = fm.height()
        painter.drawText(-w//2, h//4, text)

# --- WORKER THREADS ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    attendance_signal = pyqtSignal(str) # Emits name (for recognition) or status (for capture)
    capture_progress_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = "RECOGNITION" # "RECOGNITION", "CAPTURE", "IDLE"
        self.capture_count = 0
        self.capture_target = 30
        self.capture_dir = ""
        self.recognizer = None

    def run(self):
        if self.recognizer is None:
            self.recognizer = FaceRecognizer()

        # Camera Setup
        cap = None
        picam2 = None
        use_picamera2 = False
        
        if PICAMERA2_AVAILABLE:
            try:
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                picam2.configure(config)
                picam2.start()
                picam2.set_controls({"AeEnable": True, "AwbEnable": True})
                use_picamera2 = True
            except:
                use_picamera2 = False
        
        if not use_picamera2:
            cap = cv2.VideoCapture(0) # Default
            if not cap.isOpened():
                return 

        last_name = None
        consecutive = 0
        
        while self._run_flag:
            if use_picamera2:
                cv_img = picam2.capture_array()
            else:
                ret, cv_img = cap.read()
                if not ret: continue
            
            # Processing
            if self.mode == "RECOGNITION":
                self.process_recognition(cv_img, last_name, consecutive)
            elif self.mode == "CAPTURE":
                self.process_capture(cv_img)
            
            # Convert to Qt
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if not use_picamera2 else cv_img
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # qt_img = qt_img.scaled(800, 600, Qt.KeepAspectRatioByExpanding) # User wants responsiveness
            self.change_pixmap_signal.emit(qt_img)

        # Cleanup
        if use_picamera2: picam2.stop()
        elif cap: cap.release()

    def process_recognition(self, img, last_name, consecutive):
        # We don't draw boxes here anymore if we want a clean UI, 
        # OR we draw stylish boxes. Let's draw clean/minimal ones.
        locations, names = self.recognizer.recognize_faces(img)
        
        for (x, y, w, h), name in zip(locations, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            # Modern Corners only
            l_len = 20
            t = 2
            # Top-Left
            cv2.line(img, (x, y), (x + l_len, y), color, t)
            cv2.line(img, (x, y), (x, y + l_len), color, t)
            # Top-Right
            cv2.line(img, (x+w, y), (x+w - l_len, y), color, t)
            cv2.line(img, (x+w, y), (x+w, y + l_len), color, t)
            # Bottom-Left
            cv2.line(img, (x, y+h), (x + l_len, y+h), color, t)
            cv2.line(img, (x, y+h), (x, y+h - l_len), color, t)
            # Bottom-Right
            cv2.line(img, (x+w, y+h), (x+w - l_len, y+h), color, t)
            cv2.line(img, (x+w, y+h), (x+w, y+h - l_len), color, t)

            if name != "Unknown":
                # Logic for stable recognition
                self.attendance_signal.emit(f"MATCH:{name}")
    
    def process_capture(self, img):
         # Just detect face to ensure quality
         if self.recognizer.detector:
             h, w, _ = img.shape
             self.recognizer.detector.setInputSize((w, h))
             _, faces = self.recognizer.detector.detect(img)
             
             if faces is not None:
                 for face in faces:
                    box = face[:4].astype(int)
                    x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                    
                    # Draw a guiding circle or box
                    center_x, center_y = x + w_box//2, y + h_box//2
                    radius = int(min(w_box, h_box) / 1.5)
                    cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)
                    
                    if self.capture_count < self.capture_target:
                        self.capture_count += 1
                        filename = f"{self.capture_dir}/{self.capture_count}.jpg"
                        # Crop with margin
                        margin = 20
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(w, x + w_box + margin)
                        y2 = min(h, y + h_box + margin)
                        crop = img[y1:y2, x1:x2]
                        
                        # Save original BGR if needed, but img might be RGB from picam
                        # Let's ensure standard BGR for consistency with OpenCV
                        save_img = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if PICAMERA2_AVAILABLE else crop
                        cv2.imwrite(filename, save_img)
                        
                        # Emit Progress
                        progress = int((self.capture_count / self.capture_target) * 100)
                        self.capture_progress_signal.emit(progress)
                    else:
                        self.mode = "IDLE"
                        self.attendance_signal.emit("CAPTURE_COMPLETE")
                        break

    def start_capture(self, user_id, user_name):
        self.capture_dir = os.path.join(KNOWN_FACES_DIR, f"{user_id}_{user_name}")
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
        self.capture_count = 0
        self.mode = "CAPTURE"

    def stop(self):
        self._run_flag = False
        self.wait()
    
    def reload_model(self):
        # Trigger reload (simplified)
        self.recognizer = FaceRecognizer()

class TrainThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    def run(self):
        try:
            encoder = FaceEncoder()
            # Fake progress for UX
            # time.sleep(1) 
            success = encoder.process_images()
            if success:
                self.finished_signal.emit(True, "Success")
            else:
                self.finished_signal.emit(False, "Failed")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Access | Smart Attendance")
        self.resize(1024, 600)
        self.setStyleSheet(STYLE_MAIN)
        
        # Database
        self.db = LocalDatabase()
        
        # Central Stack
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Screens
        self.init_home_screen()
        self.init_register_screen()
        
        # Threads
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_video_feed)
        self.thread.attendance_signal.connect(self.handle_video_signal)
        self.thread.capture_progress_signal.connect(self.update_capture_progress)
        self.thread.start()
        
        self.train_thread = TrainThread()
        self.train_thread.finished_signal.connect(self.on_training_complete)

        # State
        self.last_recognized_time = 0
        
    def init_home_screen(self):
        self.home_widget = QWidget()
        layout = QHBoxLayout(self.home_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left: Video Feed Area (Full height)
        video_container = QWidget()
        video_container.setStyleSheet("background-color: black;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        video_layout.addWidget(self.video_label)
        
        # Overlay for "Welcome"
        self.overlay = OverlayLabel(self.video_label)
        self.overlay.resize(400, 80)
        self.overlay.move(120, 20) # Approximate center top
        
        # Right: Sidebar (Info Panel)
        sidebar = QFrame()
        sidebar.setFixedWidth(350)
        sidebar.setStyleSheet("background-color: #1e1e2e; border-left: 1px solid #45475a;")
        
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(20)
        side_layout.setContentsMargins(30, 50, 30, 30)
        
        # Clock
        self.lbl_time = QLabel()
        self.lbl_time.setFont(QFont("Segoe UI", 48, QFont.Bold))
        self.lbl_time.setStyleSheet("color: #89b4fa;")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        
        self.lbl_date = QLabel()
        self.lbl_date.setFont(QFont("Segoe UI", 18))
        self.lbl_date.setStyleSheet("color: #a6adc8;")
        self.lbl_date.setAlignment(Qt.AlignCenter)
        
        # Update Clock
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()
        
        side_layout.addWidget(self.lbl_time)
        side_layout.addWidget(self.lbl_date)
        
        side_layout.addStretch()
        
        # Recent Log
        lbl_recent = QLabel("Recent Activity")
        lbl_recent.setFont(QFont("Segoe UI", 14, QFont.Bold))
        side_layout.addWidget(lbl_recent)
        
        self.log_list = QListWidget()
        self.log_list.setFixedHeight(200)
        side_layout.addWidget(self.log_list)
        
        # Admin Button (Hidden/Small)
        btn_admin = QPushButton("Register User")
        btn_admin.clicked.connect(lambda: self.switch_screen(1))
        side_layout.addWidget(btn_admin)
        
        layout.addWidget(video_container, stretch=1)
        layout.addWidget(sidebar)
        
        self.central_widget.addWidget(self.home_widget)

    def init_register_screen(self):
        self.reg_widget = QWidget()
        layout = QHBoxLayout(self.reg_widget)
        
        # Left: Form
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(50, 50, 50, 50)
        form_layout.setSpacing(20)
        
        lbl_title = QLabel("New User Registration")
        lbl_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        lbl_title.setStyleSheet("color: #89b4fa;")
        
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Full Name")
        
        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Employee ID")
        
        self.btn_start = QPushButton("Start Scanning")
        self.btn_start.clicked.connect(self.start_registration)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background-color: #fab387; color: #1e1e2e;")
        self.btn_cancel.clicked.connect(self.cancel_registration)
        
        # Status / Progress
        self.progress_ring = CircularProgress()
        self.progress_ring.hide()
        
        self.lbl_status = QLabel("Ready to Scan")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        
        form_layout.addWidget(lbl_title)
        form_layout.addSpacing(20)
        form_layout.addWidget(self.input_name)
        form_layout.addWidget(self.input_id)
        form_layout.addSpacing(20)
        form_layout.addWidget(self.progress_ring, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.lbl_status)
        form_layout.addStretch()
        form_layout.addWidget(self.btn_start)
        form_layout.addWidget(self.btn_cancel)
        
        # Right: Camera Preview
        self.video_label_reg = QLabel()
        self.video_label_reg.setFixedSize(480, 640) # Portrait preview? Or square
        self.video_label_reg.setStyleSheet("background-color: black; border-radius: 20px;")
        self.video_label_reg.setScaledContents(True)

        layout.addWidget(form_container, stretch=1)
        layout.addWidget(self.video_label_reg)
        
        self.central_widget.addWidget(self.reg_widget)

    def update_clock(self):
        now = datetime.now()
        self.lbl_time.setText(now.strftime("%H:%M"))
        self.lbl_date.setText(now.strftime("%A, %d %B %Y"))

    def switch_screen(self, index):
        self.central_widget.setCurrentIndex(index)
        if index == 0:
            self.thread.mode = "RECOGNITION"
        elif index == 1:
            self.thread.mode = "IDLE" # Wait for start button

    def start_registration(self):
        name = self.input_name.text()
        uid = self.input_id.text()
        if not name or not uid:
            self.lbl_status.setText("Please enter Name and ID")
            self.lbl_status.setStyleSheet("color: #f38ba8;")
            return
        
        self.btn_start.hide()
        self.btn_cancel.hide()
        self.progress_ring.set_value(0)
        self.progress_ring.show()
        self.lbl_status.setText("Look at the camera...")
        self.lbl_status.setStyleSheet("color: #cdd6f4;")
        
        self.thread.start_capture(uid, name)

    def cancel_registration(self):
        self.thread.mode = "IDLE" # Stop capturing if running
        self.switch_screen(0)
        self.input_name.clear()
        self.input_id.clear()

    def update_video_feed(self, img):
        # Scale image to fit the label
        current_idx = self.central_widget.currentIndex()
        target_label = self.video_label if current_idx == 0 else self.video_label_reg
        
        pixmap = QPixmap.fromImage(img)
        # Scaled contents handles resizing, but let's ensure aspect ratio if needed
        # For Kiosk, filling the area is often better
        target_label.setPixmap(pixmap)

    def handle_video_signal(self, msg):
        if msg.startswith("MATCH:"):
            name = msg.split(":")[1]
            # Debounce
            now = time.time()
            if now - self.last_recognized_time > 3.0: # 3 second cooldown
                self.last_recognized_time = now
                self.show_welcome(name)
                self.log_attendance(name)
        elif msg == "CAPTURE_COMPLETE":
            self.lbl_status.setText("Processing Profile...")
            # Start Training
            self.train_thread.start()

    def update_capture_progress(self, val):
        self.progress_ring.set_value(val)
        self.lbl_status.setText(f"Scanning... {val}%")

    def show_welcome(self, name):
        self.overlay.show_message(f"Welcome, {name}!")

    def log_attendance(self, name):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_list.insertItem(0, f"✅ {name} @ {time_str}")
        self.db.add_record(DEVICE_ID, name)

    def on_training_complete(self, success, msg):
        if success:
            self.lbl_status.setText("Registration Complete!")
            self.thread.reload_model()
            QTimer.singleShot(2000, self.reset_registration)
        else:
            self.lbl_status.setText("Error: " + msg)
            self.btn_start.show()
            self.btn_cancel.show()

    def reset_registration(self):
        self.switch_screen(0)
        self.input_name.clear()
        self.input_id.clear()
        self.btn_start.show()
        self.btn_cancel.show()
        self.progress_ring.hide()
        self.lbl_status.setText("Ready")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    # Environment Fix for Raspberry Pi
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
    os.environ["XDG_SESSION_TYPE"] = "xcb"
    
    app = QApplication(sys.argv)
    window = MainApp()
    
    # Kiosk Mode Check (Optional: window.showFullScreen())
    # window.showFullScreen() 
    window.show()
    
    sys.exit(app.exec_())
