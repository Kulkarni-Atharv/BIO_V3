
import sys
import cv2
import os
import time
import shutil
import socket
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
                             QStackedWidget, QMessageBox, QFrame, QSizePolicy, 
                             QGraphicsDropShadowEffect, QListWidget, QListWidgetItem, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QIcon

# Import modules
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from core.face_encoder import FaceEncoder
from shared.config import DEVICE_ID, KNOWN_FACES_DIR, VERIFICATION_FRAMES, SERVER_IP

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
            
            # Processing - OPTIMIZATION: Process recognition every 5th frame (~8 FPS AI)
            # This keeps AI load low while allowing smoother video (40 FPS target)
            if current_mode == "RECOGNITION" and frame_count % 5 == 0:
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
            
            # Important: Prevent CPU starvation (25ms = 40 FPS target)
            self.msleep(25)

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

# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Access | Smart Attendance")
        self.resize(1024, 600)
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
        
        # NOW start the video thread after all widgets exist
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_video_feed)
        self.thread.attendance_signal.connect(self.handle_video_signal)
        self.thread.capture_progress_signal.connect(self.update_capture_progress)
        self.thread.start()
        
        self.train_thread = TrainThread()
        self.train_thread.finished_signal.connect(self.on_training_complete)

        self.last_recognized_time = 0
        
    def init_home_screen(self):
        self.home_widget = QWidget()
        layout = QHBoxLayout(self.home_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video
        video_container = QWidget()
        video_container.setStyleSheet("background-color: black;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        video_layout.addWidget(self.video_label)
        
        self.overlay = OverlayLabel(self.video_label)
        self.overlay.resize(400, 80)
        self.overlay.move(120, 20) 
        
        # Sidebar
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
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()
        
        side_layout.addWidget(self.lbl_time)
        side_layout.addWidget(self.lbl_date)
        side_layout.addStretch()
        
        lbl_recent = QLabel("Recent Activity")
        lbl_recent.setFont(QFont("Segoe UI", 14, QFont.Bold))
        side_layout.addWidget(lbl_recent)
        
        self.log_list = QListWidget()
        self.log_list.setFixedHeight(200)
        side_layout.addWidget(self.log_list)
        
        # Settings Button
        btn_settings = QPushButton(" Settings")
        # btn_settings.setIcon(QIcon("assets/settings.png")) # Usage if icon available
        btn_settings.setStyleSheet("""
            QPushButton {
                background-color: #313244; 
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        btn_settings.clicked.connect(lambda: self.switch_screen(1))
        side_layout.addWidget(btn_settings)
        
        layout.addWidget(video_container, stretch=1)
        layout.addWidget(sidebar)
        
        self.central_widget.addWidget(self.home_widget)

    def init_settings_screen(self):
        self.settings_widget = QWidget()
        main_layout = QVBoxLayout(self.settings_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
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
        btn_back.clicked.connect(lambda: self.switch_screen(0))
        
        lbl_title = QLabel("Settings")
        lbl_title.setFont(QFont("Segoe UI", 36, QFont.Bold))
        lbl_title.setStyleSheet("color: #cdd6f4;")
        
        top_bar_layout.addWidget(btn_back)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(lbl_title)
        top_bar_layout.addStretch()
        
        main_layout.addWidget(top_bar)
        
        # Simple List Menu (Mobile Style)
        menu_container = QWidget()
        menu_layout = QVBoxLayout(menu_container)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(0)
        
        # Menu Items
        add_item = self.create_menu_item("👤  Add New User", "#89b4fa", lambda: self.switch_screen(2))
        del_item = self.create_menu_item("🗑️  Delete User", "#f38ba8", self.refresh_delete_list_and_show)
        about_item = self.create_menu_item("ℹ️  About System", "#a6e3a1", self.show_about_screen)
        
        menu_layout.addWidget(add_item)
        menu_layout.addWidget(del_item)
        menu_layout.addWidget(about_item)
        menu_layout.addStretch()
        
        main_layout.addWidget(menu_container)
        self.central_widget.addWidget(self.settings_widget)
    
    def create_menu_item(self, text, accent_color, callback):
        """Create a simple mobile-style menu item"""
        item = QFrame()
        item.setFixedHeight(80)
        item.setStyleSheet(f"""
            QFrame {{
                background-color: #1e1e2e;
                border-bottom: 1px solid #45475a;
            }}
            QFrame:hover {{
                background-color: #313244;
            }}
        """)
        item.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(item)
        layout.setContentsMargins(50, 0, 50, 0)
        
        lbl_text = QLabel(text)
        lbl_text.setFont(QFont("Segoe UI", 20))
        lbl_text.setStyleSheet("color: #cdd6f4;")
        
        lbl_arrow = QLabel("→")
        lbl_arrow.setFont(QFont("Segoe UI", 24))
        lbl_arrow.setStyleSheet(f"color: {accent_color};")
        
        layout.addWidget(lbl_text)
        layout.addStretch()
        layout.addWidget(lbl_arrow)
        
        item.mousePressEvent = lambda e: callback()
        
        return item

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

    def update_clock(self):
        now = datetime.now()
        self.lbl_time.setText(now.strftime("%H:%M"))
        self.lbl_date.setText(now.strftime("%A, %d %B %Y"))

    def switch_screen(self, index):
        self.central_widget.setCurrentIndex(index)
        if index == 0:
            self.thread.set_mode("RECOGNITION")
        elif index == 2: # Register
            self.thread.set_mode("IDLE") 
        else:
            # Settings, Delete, About -> IDLE
            self.thread.set_mode("IDLE")

    def refresh_delete_list_and_show(self):
        self.delete_list.clear()
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
            target = self.video_label
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
                name = msg.split(":")[1]
                now = time.time()
                if now - self.last_recognized_time > 3.0: 
                    self.last_recognized_time = now
                    self.show_welcome(name)
                    self.log_attendance(name)
        elif current_idx == 2: # Register
             if msg == "CAPTURE_COMPLETE":
                self.lbl_status.setText("Processing Profile...")
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

if __name__ == "__main__":
    # Raspberry Pi Optimization
    if os.uname().machine.startswith('arm'):
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
