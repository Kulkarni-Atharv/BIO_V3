
import sys
import cv2
import os
import time
import shutil
import socket
import json
import ssl
import numpy as np
from datetime import datetime

import paho.mqtt.client as mqtt_client

# Try to import picamera2 for Raspberry Pi CSI cameras
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QMessageBox, QFrame, QSizePolicy, 
                             QGraphicsDropShadowEffect, QListWidget, QListWidgetItem, QGridLayout,
                             QToolButton)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QIcon

# Import modules
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from core.face_encoder import FaceEncoder
from shared.config import (
    DEVICE_ID, KNOWN_FACES_DIR, VERIFICATION_FRAMES,
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_TOPIC_RECEIVE_USERS, MQTT_TOPIC_REQUEST_USERS
)

# --- STYLESHEETS ---
STYLE_MAIN = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10px;
}
QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
    font-size: 10px;
}
QLineEdit:focus {
    border: 1px solid #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 6px;
    font-size: 11px;
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
    font-size: 16px;
    border: 1px solid #45475a;
}
QListWidget::item {
    padding: 10px;
    border-bottom: 1px solid #45475a;
}
QListWidget::item:selected {
    background-color: #45475a;
    border-radius: 5px;
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
        self.mutex = QMutex()
        self.capture_count = 0
        self.capture_target = 30
        self.capture_dir = ""
        self.recognizer = None

    def set_mode(self, mode):
        self.mutex.lock()
        self.mode = mode
        self.mutex.unlock()

    def get_mode(self):
        self.mutex.lock()
        m = self.mode
        self.mutex.unlock()
        return m

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
        frame_count = 0
        
        while self._run_flag:
            current_mode = self.get_mode()
            frame_count += 1

            if use_picamera2:
                cv_img = picam2.capture_array()
            else:
                ret, cv_img = cap.read()
                if not ret: continue
            
            # Processing - OPTIMIZATION: Process recognition every 3rd frame (approx 8-10 FPS)
            # This drastically reduces CPU load without affecting user experience.
            if current_mode == "RECOGNITION" and frame_count % 3 == 0:
                self.process_recognition(cv_img, last_name, consecutive)
            elif current_mode == "CAPTURE":
                # Capture mode needs higher FPS for smooth UI feedback
                self.process_capture(cv_img)
            
            # Convert to Qt
            # Fix Color Issue: Ensure input is treated as BGR and converted to RGB
            if use_picamera2:
                 rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                 rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            # Copy is CRITICAL for thread safety with numpy data
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)
            
            # Important: Prevent CPU starvation (40ms = 25 FPS target)
            self.msleep(40)

        # Cleanup
        if use_picamera2: picam2.stop()
        elif cap: cap.release()

    def process_recognition(self, img, last_name, consecutive):
        if self.recognizer is None:
            return
        
        # Guard against mode change mid-processing
        if self.get_mode() != "RECOGNITION":
            return

        try:
            locations, names = self.recognizer.recognize_faces(img)
        except Exception as e:
            print(f"Recognition error: {e}")
            return
        
        for (x, y, w, h), name in zip(locations, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            l_len = 20
            t = 2
            # Minimal Corners
            cv2.line(img, (x, y), (x + l_len, y), color, t)
            cv2.line(img, (x, y), (x, y + l_len), color, t)
            cv2.line(img, (x+w, y), (x+w - l_len, y), color, t)
            cv2.line(img, (x+w, y), (x+w, y + l_len), color, t)
            cv2.line(img, (x, y+h), (x + l_len, y+h), color, t)
            cv2.line(img, (x, y+h), (x, y+h - l_len), color, t)
            cv2.line(img, (x+w, y+h), (x+w - l_len, y+h), color, t)
            cv2.line(img, (x+w, y+h), (x+w, y+h - l_len), color, t)

            if name != "Unknown":
                self.attendance_signal.emit(f"MATCH:{name}")

    def process_capture(self, img):
        if self.recognizer is None or self.recognizer.detector is None:
            return
            
        try:
            h, w, _ = img.shape
            self.recognizer.detector.setInputSize((w, h))
            _, faces = self.recognizer.detector.detect(img)
            
            if faces is not None:
                for face in faces:
                   box = face[:4].astype(int)
                   x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                   
                   center_x, center_y = x + w_box//2, y + h_box//2
                   radius = int(min(w_box, h_box) / 1.5)
                   # Draw guide
                   cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)
                   
                   if self.capture_count < self.capture_target:
                       self.capture_count += 1
                       
                       # Ensure directory exists before writing
                       if not self.capture_dir or not os.path.exists(self.capture_dir):
                           # Fallback or error - but don't crash
                           print(f"Error: Capture directory missing: {self.capture_dir}")
                           self.mode = "IDLE" 
                           return

                       filename = f"{self.capture_dir}/{self.capture_count}.jpg"
                       margin = 20
                       x1 = max(0, x - margin)
                       y1 = max(0, y - margin)
                       x2 = min(w, x + w_box + margin)
                       y2 = min(h, y + h_box + margin)
                       crop = img[y1:y2, x1:x2]
                       
                       # Validate crop
                       if crop.size == 0: continue

                       save_img = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if PICAMERA2_AVAILABLE else crop
                       cv2.imwrite(filename, save_img)
                       
                       progress = int((self.capture_count / self.capture_target) * 100)
                       self.capture_progress_signal.emit(progress)
                   else:
                       self.mode = "IDLE"
                       self.attendance_signal.emit("CAPTURE_COMPLETE")
                       break
        except Exception as e:
            print(f"Capture Error: {e}")
            self.mode = "IDLE" # Reset to safe state

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
        self.recognizer = FaceRecognizer()

class TrainThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    def run(self):
        try:
            encoder = FaceEncoder()
            success = encoder.process_images()
            if success:
                self.finished_signal.emit(True, "Success")
            else:
                self.finished_signal.emit(False, "Failed")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class MQTTWorker(QThread):
    """Background thread that subscribes to receive-users and auto-updates the HMI employee list."""
    users_updated = pyqtSignal()   # Emitted after SQLite is updated — triggers UI refresh

    def __init__(self):
        super().__init__()
        self.db = LocalDatabase()
        self._stop_flag = False

    def run(self):
        client = mqtt_client.Client(client_id="hmi_user_listener", clean_session=True)
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # TLS (same as mqtt_sync.py)
        if MQTT_PORT == 8883:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            client.tls_set_context(ctx)

        def on_connect(c, userdata, flags, rc):
            if rc == 0:
                c.subscribe(MQTT_TOPIC_RECEIVE_USERS, qos=1)
                # Immediately request the employee list on connect
                c.publish(MQTT_TOPIC_REQUEST_USERS,
                          json.dumps({"device_id": DEVICE_ID, "action": "get-users"}),
                          qos=1)
            else:
                print(f"[MQTTWorker] Connect failed rc={rc}")

        def on_message(c, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                # Accept both a list and a single dict
                if isinstance(payload, dict):
                    payload = [payload]
                if not isinstance(payload, list):
                    return
                # Extract only what the CM4 needs
                stripped = [
                    {"user_id": str(u.get("user_id") or u.get("id", "")),
                     "name":    str(u.get("name")    or u.get("employee_name", ""))}
                    for u in payload
                    if (u.get("user_id") or u.get("id")) and (u.get("name") or u.get("employee_name"))
                ]
                if stripped:
                    self.db.upsert_users(stripped)
                    self.users_updated.emit()   # Tell HMI to refresh
            except Exception as e:
                print(f"[MQTTWorker] Parse error: {e}")

        client.on_connect = on_connect
        client.on_message = on_message

        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            # Keep thread alive until stop() is called
            while not self._stop_flag:
                self.msleep(500)
        except Exception as e:
            print(f"[MQTTWorker] Connection error: {e}")
        finally:
            client.loop_stop()
            client.disconnect()

    def stop(self):
        self._stop_flag = True

# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Access | Smart Attendance")
        self.resize(480, 320)
        self.setStyleSheet(STYLE_MAIN)
        
        self.db = LocalDatabase()
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # CRITICAL: Initialize ALL screens BEFORE starting video thread
        # This prevents segfaults from thread trying to update non-existent widgets
        self.init_home_screen()
        self.init_settings_screen()
        self.init_register_screen()
        self.init_delete_screen()
        self.init_about_screen()
        
        # New Screens
        self.init_user_view_screen() # 5
        self.init_user_mgt_menu()    # 6
        self.init_shift_screen()     # 7
        self.init_comm_set_menu()    # 8
        self.init_comm_params_screen() # 9
        self.init_ethernet_screen()  # 10
        self.init_wifi_screen()      # 11
        
        self.init_employee_list_screen() # 12
        
        # NOW start the video thread after all widgets exist
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_video_feed)
        self.thread.attendance_signal.connect(self.handle_video_signal)
        self.thread.capture_progress_signal.connect(self.update_capture_progress)
        self.thread.start()

        self.train_thread = TrainThread()
        self.train_thread.finished_signal.connect(self.on_training_complete)

        # MQTT worker — listens on receive-users and auto-refreshes employee list
        self.mqtt_worker = MQTTWorker()
        self.mqtt_worker.users_updated.connect(self.refresh_employee_list)
        self.mqtt_worker.start()

        self.last_recognized_time = 0
        
    def init_home_screen(self):
        self.home_widget = QWidget()
        # Use a Grid Layout to overlay controls on top of video if needed
        # But wait, video is a widget. Best way: QStackedLayout or parenting children to video_label?
        # A clean way: Main Video Widget, and overlays are children of it or siblings in a grid (0,0,1,1)
        
        main_layout = QGridLayout(self.home_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Video Background
        self.video_container = QLabel("Initializing Camera...")
        self.video_container.setAlignment(Qt.AlignCenter)
        self.video_container.setScaledContents(True)
        self.video_container.setStyleSheet("background-color: black;")
        # Add to grid at (0,0) spanning everything
        main_layout.addWidget(self.video_container, 0, 0, 4, 4) 

        # 2. Time/Date Overlay (Top Left)
        self.time_overlay = QFrame()
        self.time_overlay.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160); 
            border-radius: 8px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 30);
        """)
        # Scaled down size (was 360x140 -> ~160x60)
        self.time_overlay.setFixedSize(160, 65)
        
        time_layout = QVBoxLayout(self.time_overlay)
        time_layout.setContentsMargins(10, 5, 10, 5)
        time_layout.setSpacing(0)
        
        time_top_layout = QHBoxLayout()
        self.lbl_date_overlay = QLabel("2024-01-01")
        self.lbl_date_overlay.setFont(QFont("Segoe UI", 9))
        self.lbl_date_overlay.setStyleSheet("color: #b4befe; font-weight: bold;")
        
        self.lbl_day_overlay = QLabel("MON")
        self.lbl_day_overlay.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_day_overlay.setStyleSheet("color: #fab387;")
        self.lbl_day_overlay.setAlignment(Qt.AlignRight)
        
        time_top_layout.addWidget(self.lbl_date_overlay)
        time_top_layout.addStretch()
        time_top_layout.addWidget(self.lbl_day_overlay)
        
        self.lbl_time_overlay = QLabel("12:00:00")
        self.lbl_time_overlay.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.lbl_time_overlay.setStyleSheet("color: white;")
        self.lbl_time_overlay.setAlignment(Qt.AlignCenter)
        
        time_layout.addLayout(time_top_layout)
        time_layout.addWidget(self.lbl_time_overlay)
        
        # Add to grid Top-Left with some margin
        main_layout.addWidget(self.time_overlay, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 3. Network Status Overlay (Top Right)
        self.network_overlay = QFrame()
        self.network_overlay.setStyleSheet("""
            background-color: rgba(0, 0, 0, 100); 
            border-radius: 6px;
            color: white;
            padding: 2px;
        """)
        # Implicit size via layout
        
        net_layout = QHBoxLayout(self.network_overlay)
        net_layout.setContentsMargins(5, 2, 5, 2)
        
        self.lbl_net_icon = QLabel("🔌") # Default LAN
        self.lbl_net_icon.setFont(QFont("Segoe UI", 12))
        
        self.lbl_net_ip = QLabel("127.0.0.1")
        self.lbl_net_ip.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_net_ip.setStyleSheet("color: #a6e3a1;")
        
        net_layout.addWidget(self.lbl_net_icon)
        net_layout.addSpacing(5)
        net_layout.addWidget(self.lbl_net_ip)
        
        main_layout.addWidget(self.network_overlay, 0, 3, Qt.AlignTop | Qt.AlignRight)

        # 4. Hidden Menu Button (Transparent overlay or bottom center)
        self.btn_menu_overlay = QPushButton("⚙️") 
        self.btn_menu_overlay.setFixedSize(40, 40)
        self.btn_menu_overlay.setCursor(Qt.PointingHandCursor)
        self.btn_menu_overlay.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 100);
                color: rgba(255, 255, 255, 180);
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 200);
                color: white;
            }
        """)
        self.btn_menu_overlay.clicked.connect(lambda: self.switch_screen(1))
        # Add to Bottom-Center
        main_layout.addWidget(self.btn_menu_overlay, 3, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)
        
        # 5. Welcome Overlay (Existing)
        self.overlay = OverlayLabel(self.video_container) # Use video as parent
        self.overlay.resize(400, 80)
        self.overlay.move(120, 300) # Centered roughly
        
        # Setup Timer for Clock & Network Check
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_home_ui)
        self.timer.start(1000)
        self.update_home_ui()
        
        self.central_widget.addWidget(self.home_widget)

    def init_settings_screen(self):
        self.settings_widget = QWidget()
        main_layout = QVBoxLayout(self.settings_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top Bar
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #1e1e2e; border-bottom: 2px solid #585b70;")
        top_bar.setFixedHeight(80)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 10, 20, 10)
        
        btn_back = QPushButton("< ESC")
        btn_back.setFixedSize(100, 50)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: transparent; 
                color: #cdd6f4;
                font-size: 20px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { color: #89b4fa; }
        """)
        btn_back.clicked.connect(lambda: self.switch_screen(0))
        
        lbl_title = QLabel("MENU")
        # Match style of create_top_bar (16px Bold)
        lbl_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_title.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;") 
        lbl_title.setAlignment(Qt.AlignCenter)
        
        top_bar_layout.addWidget(btn_back)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(lbl_title)
        top_bar_layout.addStretch()
        # Add a dummy widget to balance the center alignment
        dummy = QWidget()
        dummy.setFixedSize(60, 40) # Matched size of back button roughly
        top_bar_layout.addWidget(dummy)
        
        main_layout.addWidget(top_bar)
        
        # Grid Menu Container
        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(40, 40, 40, 40)
        grid_layout.setSpacing(15)
        
        # Helper to create grid buttons
        def create_grid_btn(text, icon_emoji, row, col, callback=None):
            btn = QToolButton()
            btn.setText(f"{icon_emoji}\n{text}")
            btn.setFont(QFont("Segoe UI", 16, QFont.Bold))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setCursor(Qt.PointingHandCursor)
            
            # Replicating the blue tile style
            btn.setStyleSheet("""
                QToolButton {
                    background-color: #0078d7; 
                    color: white;
                    border: none;
                    border-radius: 0px; 
                    padding: 10px;
                    font-size: 18px;
                }
                QToolButton:hover {
                    background-color: #0063b1;
                }
                QToolButton:pressed {
                    background-color: #005a9e;
                }
            """)
            
            # Using emojis as icons roughly matching the image
            # Ideally we'd use QIcon with actual resource files
            
            if callback:
                btn.clicked.connect(callback)
            
            grid_layout.addWidget(btn, row, col)
            return btn

        # Row 0
        create_grid_btn("User Mgt", "👥", 0, 0, lambda: self.switch_screen(6))
        create_grid_btn("Shift", "📅", 0, 1, lambda: self.switch_screen(7))
        
        # Row 1
        create_grid_btn("Comm set", "⚙️", 1, 0, lambda: self.switch_screen(8))
        create_grid_btn("Sys info", "ℹ️", 1, 1, self.show_about_screen)

        main_layout.addWidget(grid_container)
        self.central_widget.addWidget(self.settings_widget)

    def handle_user_mgt(self):
        # Allow choosing between Add and Delete since "User Mgt" usually implies both
        # For now, default to Add User screen (2)
        self.switch_screen(2)

    def show_info_toast(self, message):
        QMessageBox.information(self, "Info", message)
    
    # Simple placeholder for create_menu_item to avoid breaking potential other calls if any (though none seen)
    def create_menu_item(self, text, accent_color, callback):
        return QWidget()

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
        
        self.btn_cancel_reg = QPushButton("Cancel")
        self.btn_cancel_reg.setStyleSheet("background-color: #fab387; color: #1e1e2e;")
        self.btn_cancel_reg.clicked.connect(lambda: self.switch_screen(1))
        
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
        form_layout.addWidget(self.btn_cancel_reg)
        
        # Right: Camera Preview
        self.video_label_reg = QLabel()
        self.video_label_reg.setFixedSize(480, 640)
        self.video_label_reg.setStyleSheet("background-color: black; border-radius: 20px;")
        self.video_label_reg.setScaledContents(True)

        layout.addWidget(form_container, stretch=1)
        layout.addWidget(self.video_label_reg)
        
        self.central_widget.addWidget(self.reg_widget)

    def init_delete_screen(self):
        self.del_widget = QWidget()
        main_layout = QVBoxLayout(self.del_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top Bar
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #1e1e2e; border-bottom: 1px solid #45475a;")
        top_bar.setFixedHeight(100)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(50, 20, 50, 20)
        
        btn_back = QPushButton("← Back")
        btn_back.setFixedSize(120, 50)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_back.clicked.connect(lambda: self.switch_screen(1))
        
        lbl_title = QLabel("Delete User")
        lbl_title.setFont(QFont("Segoe UI", 36, QFont.Bold))
        lbl_title.setStyleSheet("color: #f38ba8;")
        
        top_bar_layout.addWidget(btn_back)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(lbl_title)
        top_bar_layout.addStretch()
        
        main_layout.addWidget(top_bar)
        
        # Content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(80, 40, 80, 40)
        content_layout.setSpacing(20)
        
        lbl_instruction = QLabel("Select a user to remove from the system")
        lbl_instruction.setFont(QFont("Segoe UI", 16))
        lbl_instruction.setStyleSheet("color: #a6adc8;")
        content_layout.addWidget(lbl_instruction)
        
        self.delete_list = QListWidget()
        self.delete_list.setFont(QFont("Segoe UI", 18))
        self.delete_list.setStyleSheet("""
            QListWidget {
                background-color: #313244;
                border-radius: 15px;
                padding: 15px;
            }
            QListWidget::item {
                padding: 15px;
                border-bottom: 1px solid #45475a;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background-color: #45475a;
            }
            QListWidget::item:hover {
                background-color: #3a3a4a;
            }
        """)
        content_layout.addWidget(self.delete_list)
        
        btn_confirm_del = QPushButton("Delete Selected User")
        btn_confirm_del.setFixedHeight(60)
        btn_confirm_del.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                border-radius: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """)
        btn_confirm_del.clicked.connect(self.delete_selected_user)
        content_layout.addWidget(btn_confirm_del)
        
        main_layout.addWidget(content)
        self.central_widget.addWidget(self.del_widget)

    def init_about_screen(self):
        self.about_widget = QWidget()
        main_layout = QVBoxLayout(self.about_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top Bar
        top_bar = QFrame()
        top_bar.setStyleSheet("background-color: #1e1e2e; border-bottom: 1px solid #45475a;")
        top_bar.setFixedHeight(100)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(50, 20, 50, 20)
        
        btn_back = QPushButton("← Back")
        btn_back.setFixedSize(120, 50)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_back.clicked.connect(lambda: self.switch_screen(1))
        
        lbl_title = QLabel("About System")
        lbl_title.setFont(QFont("Segoe UI", 36, QFont.Bold))
        lbl_title.setStyleSheet("color: #a6e3a1;")
        
        top_bar_layout.addWidget(btn_back)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(lbl_title)
        top_bar_layout.addStretch()
        
        main_layout.addWidget(top_bar)
        
        # Content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(80, 60, 80, 60)
        content_layout.setSpacing(30)
        
        # Info Cards
        ip_card = self.create_info_card("Network Address", "Loading...", "#89b4fa")
        self.lbl_ip = ip_card.findChild(QLabel, "value_label")
        
        dev_card = self.create_info_card("Device ID", DEVICE_ID, "#a6e3a1")
        ver_card = self.create_info_card("Software Version", "2.0.0 (Kiosk Edition)", "#f9e2af")
        
        content_layout.addWidget(ip_card)
        content_layout.addWidget(dev_card)
        content_layout.addWidget(ver_card)
        content_layout.addStretch()
        
        main_layout.addWidget(content)
        self.central_widget.addWidget(self.about_widget)
    
    def create_info_card(self, label, value, accent_color):
        """Create a professional info display card"""
        card = QFrame()
        card.setFixedHeight(120)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #313244;
                border-radius: 15px;
                border-left: 5px solid {accent_color};
            }}
        """)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 20, 30, 20)
        card_layout.setSpacing(10)
        
        lbl_label = QLabel(label)
        lbl_label.setFont(QFont("Segoe UI", 14))
        lbl_label.setStyleSheet("color: #a6adc8;")
        
        lbl_value = QLabel(value)
        lbl_value.setObjectName("value_label")
        lbl_value.setFont(QFont("Segoe UI", 22, QFont.Bold))
        lbl_value.setStyleSheet(f"color: {accent_color};")
        
        card_layout.addWidget(lbl_label)
        card_layout.addWidget(lbl_value)
        
        return card

    def update_home_ui(self):
        # Update Time
        now = datetime.now()
        self.lbl_time_overlay.setText(now.strftime("%H:%M:%S"))
        self.lbl_date_overlay.setText(now.strftime("%Y-%m-%d"))
        self.lbl_day_overlay.setText(now.strftime("%a").upper())
        
        # Update Network (Every 5 seconds roughly or just check quickly)
        # Optimization: Only check every 5th second
        if int(now.timestamp()) % 5 == 0:
            self.check_network_status()

    def check_network_status(self):
        ip = "127.0.0.1"
        icon = "❌" # Disconnected
        color = "#f38ba8" 
        
        try:
            # Simple check
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Try connecting to Google DNS to get external facing IP
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            
            # Simple Heuristic for Icon (Linux specific mostly)
            # On Windows, hard to tell without psutil.
            # Assuming if IP exists -> Connected.
            # Default to LAN icon if we can't tell.
            icon = "🔌" # LAN
            
            # Try to guess WiFi based on common interface names if on Linux
            if os.path.exists("/proc/net/wireless"):
                with open("/proc/net/wireless", "r") as f:
                    if "wlan" in f.read():
                         icon = "📶" # WiFi
                         
            color = "#a6e3a1" # Green
            
        except:
            ip = "Disconnected"
            icon = "❌"
            color = "#f38ba8"

        self.lbl_net_ip.setText(ip)
        self.lbl_net_icon.setText(icon)
        self.lbl_net_ip.setStyleSheet(f"color: {color};")

    def switch_screen(self, index):
        self.central_widget.setCurrentIndex(index)
        if index == 0:
            self.thread.set_mode("RECOGNITION")
        elif index == 2:  # Register
            self.thread.set_mode("IDLE")
        elif index == 12: # Employee List — always refresh on open
            self.refresh_employee_list()
        else:
            self.thread.set_mode("IDLE")

    # --- NEW MENUS ---
    def init_user_mgt_menu(self):
        self.user_mgt_widget = QWidget()
        layout = QVBoxLayout(self.user_mgt_widget)
        
        # Top Bar
        top_bar = self.create_top_bar("User Management", lambda: self.switch_screen(1))
        layout.addWidget(top_bar)
        
        # Grid
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setContentsMargins(50, 50, 50, 50)
        grid.setSpacing(20)
        
        self.create_grid_btn(grid, "Add User",       "👤", 0, 0, lambda: self.switch_screen(2))
        self.create_grid_btn(grid, "Employee List",  "📋", 0, 1, lambda: self.switch_screen(12))
        self.create_grid_btn(grid, "User View",      "👀", 1, 0, lambda: self.refresh_user_view_and_show())
        self.create_grid_btn(grid, "Delete User",    "🗑️", 1, 1, self.refresh_delete_list_and_show)
        
        layout.addWidget(grid_container)
        self.central_widget.addWidget(self.user_mgt_widget)

    def init_user_view_screen(self):
        self.user_view_widget = QWidget()
        layout = QVBoxLayout(self.user_view_widget)
        
        layout.addWidget(self.create_top_bar("User List", lambda: self.switch_screen(6)))
        
        self.user_list_view = QListWidget()
        self.user_list_view.setStyleSheet("""
            QListWidget { background-color: #313244; border-radius: 10px; padding: 10px; font-size: 18px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #45475a; }
        """)
        layout.addWidget(self.user_list_view)
        
        self.central_widget.addWidget(self.user_view_widget)

    def init_shift_screen(self):
        self.shift_widget = QWidget()
        layout = QVBoxLayout(self.shift_widget)
        
        layout.addWidget(self.create_top_bar("Shift Management", lambda: self.switch_screen(1)))
        
        form = QWidget()
        form_layout = QGridLayout(form) 
        form_layout.setContentsMargins(100, 50, 100, 50)
        form_layout.setSpacing(30)
        
        lbl_start = QLabel("Shift Start Time:")
        lbl_start.setFont(QFont("Segoe UI", 18))
        self.input_shift_start = QLineEdit("09:00")
        
        lbl_end = QLabel("Shift End Time:")
        lbl_end.setFont(QFont("Segoe UI", 18))
        self.input_shift_end = QLineEdit("18:00")
        
        btn_save = QPushButton("Save Shift")
        btn_save.clicked.connect(lambda: QMessageBox.information(self, "Success", "Shift Updated!"))
        
        form_layout.addWidget(lbl_start, 0, 0)
        form_layout.addWidget(self.input_shift_start, 0, 1)
        form_layout.addWidget(lbl_end, 1, 0)
        form_layout.addWidget(self.input_shift_end, 1, 1)
        form_layout.addWidget(btn_save, 2, 1)
        
        layout.addWidget(form)
        layout.addStretch()
        self.central_widget.addWidget(self.shift_widget)

    def init_comm_set_menu(self):
        self.comm_menu_widget = QWidget()
        layout = QVBoxLayout(self.comm_menu_widget)
        
        layout.addWidget(self.create_top_bar("Communication", lambda: self.switch_screen(1)))
        
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setContentsMargins(50, 50, 50, 50)
        grid.setSpacing(20)
        
        self.create_grid_btn(grid, "Comm Params", "⚙️", 0, 0, lambda: self.switch_screen(9))
        self.create_grid_btn(grid, "Ethernet", "🌐", 0, 1, lambda: self.switch_screen(10))
        self.create_grid_btn(grid, "WIFI", "📶", 1, 0, lambda: self.switch_screen(11))
        
        layout.addWidget(grid_container)
        self.central_widget.addWidget(self.comm_menu_widget)

    def init_comm_params_screen(self):
        self.comm_params_widget = QWidget()
        layout = QVBoxLayout(self.comm_params_widget)
        layout.addWidget(self.create_top_bar("Comm Params", lambda: self.switch_screen(8)))
        
        form = QWidget()
        form_layout = QGridLayout(form)
        form_layout.setContentsMargins(100, 50, 100, 50)
        
        self.input_dev_id = QLineEdit(str(DEVICE_ID))
        self.input_port = QLineEdit("8080")
        
        form_layout.addWidget(QLabel("Device ID:"), 0, 0)
        form_layout.addWidget(self.input_dev_id, 0, 1)
        form_layout.addWidget(QLabel("Port No:"), 1, 0)
        form_layout.addWidget(self.input_port, 1, 1)
        
        layout.addWidget(form)
        layout.addStretch()
        self.central_widget.addWidget(self.comm_params_widget)

    def init_ethernet_screen(self):
        self.eth_widget = QWidget()
        layout = QVBoxLayout(self.eth_widget)
        layout.addWidget(self.create_top_bar("Ethernet Settings", lambda: self.switch_screen(8)))
        
        form = QWidget()
        form_layout = QGridLayout(form)
        form_layout.setContentsMargins(50, 20, 50, 20)
        form_layout.setSpacing(15)
        
        # Helper to add row
        def add_row(label, val, row):
            l = QLabel(label)
            l.setFont(QFont("Segoe UI", 16))
            i = QLineEdit(val)
            form_layout.addWidget(l, row, 0)
            form_layout.addWidget(i, row, 1)
            return i
            
        self.input_ip = add_row("IP Address:", "192.168.1.100", 0)
        self.input_subnet = add_row("Subnet Mask:", "255.255.255.0", 1)
        self.input_gateway = add_row("Gateway:", "192.168.1.1", 2)
        self.input_dns = add_row("DNS Server:", "8.8.8.8", 3)
        
        # MAC Read only
        l_mac = QLabel("MAC Address:")
        l_mac.setFont(QFont("Segoe UI", 16))
        self.lbl_mac = QLabel("aa:bb:cc:dd:ee:ff")
        self.lbl_mac.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 18px;")
        form_layout.addWidget(l_mac, 4, 0)
        form_layout.addWidget(self.lbl_mac, 4, 1)
        
        layout.addWidget(form)
        layout.addStretch()
        self.central_widget.addWidget(self.eth_widget)

    def init_wifi_screen(self):
        self.wifi_widget = QWidget()
        layout = QVBoxLayout(self.wifi_widget)
        layout.addWidget(self.create_top_bar("WiFi Settings", lambda: self.switch_screen(8)))
        
        lbl = QLabel("WiFi Scanning not implemented yet.")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        
        self.central_widget.addWidget(self.wifi_widget)

    # --- HELPERS ---
    def create_top_bar(self, title, back_callback):
        frame = QFrame()
        frame.setStyleSheet("background-color: #1e1e2e; border-bottom: 2px solid #585b70;")
        # Reduced height for 320px height screen
        frame.setFixedHeight(40)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        btn = QPushButton("<")
        btn.setStyleSheet("background-color: transparent; color: #cdd6f4; font-size: 14px; border: none; font-weight: bold;")
        btn.clicked.connect(back_callback)
        
        lbl = QLabel(title)
        lbl.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(btn)
        layout.addWidget(lbl, stretch=1)
        # Dummy for balance
        d = QWidget(); d.setFixedSize(20, 10); layout.addWidget(d)
        
        return frame

    def create_grid_btn(self, layout, text, icon, row, col, callback):
        btn = QToolButton()
        # Scaled down fonts
        btn.setText(f"{icon}\n{text}")
        btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QToolButton { background-color: #0078d7; color: white; border: none; padding: 5px; font-size: 12px; }
            QToolButton:hover { background-color: #0063b1; }
        """)
        if callback:
            btn.clicked.connect(callback)
        layout.addWidget(btn, row, col)

    def refresh_user_view_and_show(self):
        self.user_list_view.clear()
        if os.path.exists(KNOWN_FACES_DIR):
            users = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
            for user in users:
                self.user_list_view.addItem(QListWidgetItem(user))
        self.switch_screen(5)

    def refresh_delete_list_and_show(self):
        self.delete_list.clear() # Fix for existing function needing update
        if os.path.exists(KNOWN_FACES_DIR):
            users = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
            for user in users:
                self.delete_list.addItem(QListWidgetItem(user))
        self.switch_screen(3)

    def delete_selected_user(self):
        item = self.delete_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Selection", "Please select a user to delete.")
            return
        
        user_dir = item.text()
        confirm = QMessageBox.question(self, "Confirm Delete", 
                                     f"Are you sure you want to delete '{user_dir}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            full_path = os.path.join(KNOWN_FACES_DIR, user_dir)
            try:
                shutil.rmtree(full_path)
                QMessageBox.information(self, "Success", f"User '{user_dir}' deleted.")
                self.refresh_delete_list_and_show()
                # Trigger model reload
                self.train_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")

    def show_about_screen(self):
        # Get IP
        ip = "Unknown"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except:
            ip = "127.0.0.1"
        
        self.lbl_ip.setText(f"IP Address: {ip}")
        self.switch_screen(4)

    def start_registration(self):
        name = self.input_name.text()
        uid = self.input_id.text()
        if not name or not uid:
            self.lbl_status.setText("Enter Name and ID")
            self.lbl_status.setStyleSheet("color: #f38ba8;")
            return
        
        self.btn_start.hide()
        self.btn_cancel_reg.hide()
        self.progress_ring.set_value(0)
        self.progress_ring.show()
        self.lbl_status.setText("Look at the camera...")
        self.lbl_status.setStyleSheet("color: #cdd6f4;")
        
        self.thread.start_capture(uid, name)

    def update_video_feed(self, img):
        current_idx = self.central_widget.currentIndex()
        # Only show video in Home(0) and Register(2)
        if current_idx == 0:
            target = self.video_container
        elif current_idx == 2:
            target = self.video_label_reg
        else:
            return
        
        try:
            pixmap = QPixmap.fromImage(img)
            target.setPixmap(pixmap)
        except:
            # Silently ignore any Qt errors during screen transitions
            pass

    def handle_video_signal(self, msg):
        current_idx = self.central_widget.currentIndex()
        if current_idx == 0: # Home
            if msg.startswith("MATCH:"):
                # msg format: "MATCH:ID_Name" or "MATCH:Name"
                # If FaceEncoder uses folder name, it could be "101_Atharv"
                full_identity = msg.split("MATCH:")[1]
                
                user_id = full_identity
                name = full_identity
                
                # Check if formatted as ID_Name
                # Simple check: digits followed by underscore
                # Or just split by first underscore
                if "_" in full_identity:
                    parts = full_identity.split('_', 1) 
                    # Attempt to see if first part is ID-like? 
                    # Actually, let's just assume strict "ID_Name" format for simplicity if underscore exists
                    user_id = parts[0]
                    name = parts[1]

                now = time.time()
                if now - self.last_recognized_time > 3.0: 
                    self.last_recognized_time = now
                    self.show_welcome(name)
                    self.log_attendance(user_id, name)
        elif current_idx == 2: # Register
             if msg == "CAPTURE_COMPLETE":
                self.lbl_status.setText("Processing Profile...")
                self.train_thread.start()

    def update_capture_progress(self, val):
        self.progress_ring.set_value(val)
        self.lbl_status.setText(f"Scanning... {val}%")

    def show_welcome(self, name):
        self.overlay.show_message(f"Welcome, {name}!")

    def log_attendance(self, user_id, name):
        # time_str removed as it's handled in DB
        # Call updated add_record with user_id
        # Confidence is not passed from Recognizer yet, default to 0.0 or update recognizer later
        self.db.add_record(DEVICE_ID, name, user_id=user_id)

    def on_training_complete(self, success, msg):
        if self.central_widget.currentIndex() == 2: # Register Mode
            if success:
                self.lbl_status.setText("Registration Complete!")
                self.thread.reload_model()
                QTimer.singleShot(2000, self.reset_registration)
            else:
                self.lbl_status.setText("Error: " + msg)
                self.btn_start.show()
                self.btn_cancel_reg.show()
        else:
             # Likely background update from delete
             if success:
                 self.thread.reload_model()

    def reset_registration(self):
        self.switch_screen(1) # Back to Settings
        self.input_name.clear()
        self.input_id.clear()
        self.btn_start.show()
        self.btn_cancel_reg.show()
        self.progress_ring.hide()
        self.lbl_status.setText("Ready")
        self.thread.set_mode("IDLE")  # Ensure we stop scanning when resetting

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def init_employee_list_screen(self):
        """Screen 12 — Employee list from dashboard with face-registration status."""
        self.emp_list_widget = QWidget()
        layout = QVBoxLayout(self.emp_list_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        top_bar = self.create_top_bar("Employee List", lambda: self.switch_screen(6))
        layout.addWidget(top_bar)

        # Hint label
        hint = QLabel("Tap ⚠️ row to register face")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #f9e2af; font-size: 11px; padding: 2px;")
        layout.addWidget(hint)

        # List widget
        self.emp_list_view = QListWidget()
        self.emp_list_view.setFont(QFont("Segoe UI", 13))
        self.emp_list_view.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                border: none;
                padding: 4px;
            }
            QListWidget::item {
                background-color: #313244;
                border-radius: 6px;
                margin: 3px 6px;
                padding: 8px 10px;
                border-left: 4px solid #45475a;
            }
            QListWidget::item:selected {
                background-color: #45475a;
            }
            QListWidget::item:hover {
                background-color: #3a3a4a;
            }
        """)
        self.emp_list_view.itemClicked.connect(self.on_employee_item_clicked)
        layout.addWidget(self.emp_list_view)

        # Bottom refresh button
        btn_refresh = QPushButton("🔄  Refresh List")
        btn_refresh.setFixedHeight(36)
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 0px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45475a; }
        """)
        btn_refresh.clicked.connect(self.refresh_employee_list)
        layout.addWidget(btn_refresh)

        self.central_widget.addWidget(self.emp_list_widget)

    def refresh_employee_list(self):
        """Reload employee list from SQLite and mark registration status."""
        self.emp_list_view.clear()

        # Registered face folders: 'user_id_name' or just 'name'
        registered_ids = set()
        if os.path.exists(KNOWN_FACES_DIR):
            for folder in os.listdir(KNOWN_FACES_DIR):
                if os.path.isdir(os.path.join(KNOWN_FACES_DIR, folder)):
                    # Try to extract user_id from folder name 'ID_Name'
                    registered_ids.add(folder.split('_')[0] if '_' in folder else folder)

        users = self.db.get_all_users()

        if not users:
            item = QListWidgetItem("  No employees found. Sync from dashboard first.")
            item.setForeground(QColor("#a6adc8"))
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.emp_list_view.addItem(item)
            return

        for u in users:
            uid  = u["user_id"]
            name = u["name"]
            is_registered = (uid in registered_ids)

            if is_registered:
                badge = "✅"
                color = "#a6e3a1"   # green
                left_border = "#a6e3a1"
            else:
                badge = "⚠️"
                color = "#f9e2af"   # yellow
                left_border = "#f9e2af"

            label = f"  {badge}  {uid:<8}  {name}"
            item  = QListWidgetItem(label)
            item.setForeground(QColor(color))
            # Store user data for click handler
            item.setData(Qt.UserRole, {"user_id": uid, "name": name, "registered": is_registered})
            # Colour the left border via stylesheet on item isn't directly possible —
            # we differentiate only by foreground colour
            self.emp_list_view.addItem(item)

        # Status summary at bottom
        total = len(users)
        reg_count = sum(1 for u in users if u["user_id"] in registered_ids)
        summary = QListWidgetItem(f"  📊  {reg_count}/{total} registered")
        summary.setForeground(QColor("#89b4fa"))
        summary.setFlags(summary.flags() & ~Qt.ItemIsSelectable)
        self.emp_list_view.addItem(summary)

    def on_employee_item_clicked(self, item):
        """Pre-fill name/ID and jump to face capture screen for unregistered users."""
        data = item.data(Qt.UserRole)
        if not data:
            return  # Summary row or info row

        uid  = data["user_id"]
        name = data["name"]
        is_registered = data["registered"]

        if is_registered:
            QMessageBox.information(
                self, "Already Registered",
                f"{name} ({uid}) already has a registered face.\n"
                "Delete the existing entry first to re-register."
            )
            return

        # Pre-fill the registration form and go to capture screen
        self.input_name.setText(name)
        self.input_id.setText(uid)
        self.lbl_status.setText("Ready to Scan")
        self.lbl_status.setStyleSheet("color: #cdd6f4;")
        self.btn_start.show()
        self.btn_cancel_reg.show()
        self.progress_ring.hide()
        self.switch_screen(2)  # Go to Register screen (index 2)

    def closeEvent(self, event):
        self.thread.stop()
        self.mqtt_worker.stop()
        self.mqtt_worker.wait()
        event.accept()


if __name__ == "__main__":
    # Raspberry Pi Optimization (Platform Check)
    import platform
    if platform.system() == "Linux" and platform.machine().startswith('arm'):
        # os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
        os.environ["XDG_SESSION_TYPE"] = "xcb"
    
    # Global Exception Hook to catch crashes
    def exception_hook(exctype, value, traceback):
        print(f"CRITICAL ERROR: {exctype}, {value}")
        sys.__excepthook__(exctype, value, traceback)
        sys.exit(1)
        
    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    
    # Font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    try:
        window = MainApp()
        # window.showFullScreen() 
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application Crashed: {e}")

