
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

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QBoxLayout, QLabel, QPushButton, QLineEdit,
                             QStackedWidget, QMessageBox, QFrame, QSizePolicy,
                             QListWidget, QListWidgetItem, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QMutex, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen

# Import modules
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from core.face_encoder import FaceEncoder
from core.face_guide import RegistrationGuide, crop_face, MIN_VALID_SAMPLES
from device.camera import open_usb_camera
from device.relay import MachineRelay
from shared.config import (
    DEVICE_ID, KNOWN_FACES_DIR, NAMES_FILE, VERIFICATION_FRAMES, MACHINE_OUTPUT,
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_TOPIC_RECEIVE_USERS, MQTT_TOPIC_REQUEST_USERS
)

# --- THEME (dark industrial HMI, sized for a 5" 1280x720 touch panel) ---
C_BG      = "#11111b"   # window background
C_PANEL   = "#181825"   # panels / bars
C_CARD    = "#1e1e2e"   # cards / tiles
C_RAISED  = "#313244"   # buttons, inputs
C_BORDER  = "#45475a"
C_TEXT    = "#cdd6f4"
C_MUTED   = "#a6adc8"
C_BLUE    = "#89b4fa"
C_GREEN   = "#a6e3a1"
C_YELLOW  = "#f9e2af"
C_RED     = "#f38ba8"

SCAN_TIMEOUT_S = 10      # give up a scan after this many seconds
RESULT_HOLD_MS = 4000    # how long the result card stays on screen

STYLE_MAIN = f"""
QMainWindow {{
    background-color: {C_BG};
}}
QLabel {{
    color: {C_TEXT};
}}
QLineEdit {{
    background-color: {C_RAISED};
    color: {C_TEXT};
    border: 2px solid {C_BORDER};
    border-radius: 10px;
    padding: 8px 14px;
    min-height: 40px;
    font-size: 22px;
}}
QLineEdit:focus {{
    border: 2px solid {C_BLUE};
}}
QLineEdit:disabled {{
    color: {C_MUTED};
}}
QPushButton {{
    background-color: {C_BLUE};
    color: {C_BG};
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 20px;
    font-weight: bold;
}}
QPushButton:pressed {{
    background-color: #74c7ec;
}}
QPushButton:disabled {{
    background-color: {C_RAISED};
    color: #6c7086;
}}
QListWidget {{
    background-color: {C_CARD};
    border-radius: 12px;
    padding: 8px;
    color: {C_TEXT};
    font-size: 20px;
    border: 1px solid {C_RAISED};
}}
QListWidget::item {{
    padding: 14px;
    border-bottom: 1px solid {C_RAISED};
}}
QListWidget::item:selected {{
    background-color: {C_RAISED};
    border-radius: 8px;
}}
QScrollBar:vertical {{
    background: {C_PANEL};
    width: 28px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    min-height: 60px;
    border-radius: 12px;
    margin: 3px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QMessageBox {{
    background-color: {C_CARD};
}}
QMessageBox QLabel {{
    font-size: 20px;
    min-width: 360px;
}}
QMessageBox QPushButton {{
    min-width: 140px;
}}
"""

# --- CUSTOM WIDGETS ---
class ScanButton(QPushButton):
    """Large round START button that turns into a progress ring while scanning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        # Own stylesheet so the global QPushButton min-height/padding can't shrink it
        self.setStyleSheet("QPushButton { min-width: 300px; max-width: 300px; min-height: 300px;"
                           " max-height: 300px; padding: 0; border: none; background: transparent; }")
        self.setCursor(Qt.PointingHandCursor)
        self.scanning = False
        self.stop_mode = False  # True while the machine is ON: red STOP button
        self.progress = 0.0     # 0..1 while scanning
        self.subtitle = "Tap to start machine"

    def set_ready(self):
        self.scanning = False
        self.stop_mode = False
        self.progress = 0.0
        self.subtitle = "Tap to start machine"
        self.setEnabled(True)
        self.update()

    def set_stop(self):
        self.scanning = False
        self.stop_mode = True
        self.progress = 0.0
        self.subtitle = "Tap to stop machine"
        self.setEnabled(True)
        self.update()

    def set_scanning(self, progress):
        self.scanning = True
        self.stop_mode = False
        self.progress = max(0.0, min(1.0, progress))
        self.subtitle = f"{max(0, int(SCAN_TIMEOUT_S * (1 - self.progress) + 0.99))} s"
        self.setEnabled(False)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        outer = self.rect().adjusted(8, 8, -8, -8)
        inner = self.rect().adjusted(30, 30, -30, -30)

        if self.scanning:
            # Track + progress arc
            p.setPen(QPen(QColor(C_RAISED), 16))
            p.drawEllipse(outer)
            p.setPen(QPen(QColor(C_BLUE), 16, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(outer, 90 * 16, -int(360 * 16 * self.progress))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C_CARD))
            p.drawEllipse(inner)
            title, title_color, sub_color = "SCANNING", C_BLUE, C_MUTED
        elif self.stop_mode:
            # Soft halo + solid red STOP button
            p.setPen(QPen(QColor(243, 139, 168, 70), 16))
            p.drawEllipse(outer)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#e57a98") if self.isDown() else QColor(C_RED))
            p.drawEllipse(inner)
            title, title_color, sub_color = "STOP", C_BG, "#5c2333"
        else:
            # Soft halo + solid green button
            p.setPen(QPen(QColor(166, 227, 161, 70), 16))
            p.drawEllipse(outer)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#8fd18a") if self.isDown() else QColor(C_GREEN))
            p.drawEllipse(inner)
            title, title_color, sub_color = "START", C_BG, "#2f5131"

        f = QFont(self.font())
        f.setPixelSize(52 if not self.scanning else 34)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(title_color))
        p.drawText(inner.adjusted(0, -30, 0, -30), Qt.AlignCenter, title)

        f.setPixelSize(20)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(sub_color))
        p.drawText(inner.adjusted(0, 50, 0, 50), Qt.AlignCenter, self.subtitle)

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

class VirtualKeyboard(QFrame):
    """On-screen keyboard for the touch panel. Types into whichever QLineEdit has focus."""

    ROWS = [
        list("1234567890"),
        list("qwertyuiop"),
        list("asdfghjkl"),
        ["SHIFT"] + list("zxcvbnm") + ["BACK"],
        ["SPACE", ".", "-", ":", "DONE"],
    ]
    STRETCH = {"SHIFT": 3, "BACK": 3, "SPACE": 12, "DONE": 4}
    LABELS = {"SHIFT": "⇧", "BACK": "⌫", "SPACE": "space", "DONE": "DONE"}
    KEY_H = 54

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("vkb")
        self.setStyleSheet(
            f"#vkb {{ background-color: {C_PANEL}; border-top: 2px solid {C_BORDER}; }}"
            f"QPushButton {{ background-color: {C_RAISED}; color: {C_TEXT}; border-radius: 10px;"
            f" padding: 0; font-size: 24px; font-weight: normal; }}"
            f"QPushButton:pressed {{ background-color: {C_BORDER}; }}"
            f"QPushButton[special='true'] {{ background-color: {C_CARD}; color: {C_MUTED}; font-size: 22px; }}"
            f"QPushButton[special='true']:pressed {{ background-color: {C_BORDER}; }}"
            f"QPushButton[active='true'] {{ background-color: {C_BLUE}; color: {C_BG}; }}"
            f"QPushButton[done='true'] {{ background-color: {C_GREEN}; color: {C_BG}; font-weight: bold; }}")
        self.shift = False
        self.letter_keys = []
        self.shift_key = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        for i, row in enumerate(self.ROWS):
            h = QHBoxLayout()
            h.setSpacing(6)
            if i == 2:
                h.addStretch(1)   # stagger the home row like a real keyboard
            for key in row:
                h.addWidget(self._make_key(key), stretch=self.STRETCH.get(key, 2))
            if i == 2:
                h.addStretch(1)
            layout.addLayout(h)
        self.setFixedHeight(self.sizeHint().height())
        self.hide()

    def _make_key(self, key):
        btn = QPushButton(self.LABELS.get(key, key))
        btn.setFocusPolicy(Qt.NoFocus)   # keep focus (and the cursor) in the text box
        btn.setFixedHeight(self.KEY_H)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if key in ("SHIFT", "BACK", "SPACE"):
            btn.setProperty("special", True)
        if key == "DONE":
            btn.setProperty("done", True)
        if key == "SHIFT":
            self.shift_key = btn
        if len(key) == 1 and key.isalpha():
            self.letter_keys.append(btn)
        btn.clicked.connect(lambda _, k=key: self._press(k))
        return btn

    def _set_shift(self, on):
        self.shift = on
        for btn in self.letter_keys:
            btn.setText(btn.text().upper() if on else btn.text().lower())
        self.shift_key.setProperty("active", on)
        self.shift_key.style().unpolish(self.shift_key)
        self.shift_key.style().polish(self.shift_key)

    def _press(self, key):
        if key == "DONE":
            self.close_keyboard()
            return
        if key == "SHIFT":
            self._set_shift(not self.shift)
            return
        target = QApplication.focusWidget()
        if not isinstance(target, QLineEdit):
            return
        if key == "BACK":
            target.backspace()
        elif key == "SPACE":
            target.insert(" ")
            self._set_shift(True)    # capitalise the next word (names)
        else:
            target.insert(key.upper() if self.shift else key)
            if self.shift:
                self._set_shift(False)

    def show_for(self, line_edit):
        # Capitalise the first letter when the box is empty
        self._set_shift(not line_edit.text())
        self.place()
        self.show()
        self.raise_()

    def place(self):
        parent = self.parentWidget()
        self.setGeometry(0, parent.height() - self.height(), parent.width(), self.height())

    def close_keyboard(self):
        self.hide()
        target = QApplication.focusWidget()
        if isinstance(target, QLineEdit):
            target.clearFocus()

# --- WORKER THREADS ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    attendance_signal = pyqtSignal(str) # Emits name (for recognition) or status (for capture)
    capture_progress_signal = pyqtSignal(int)
    guidance_signal = pyqtSignal(str, str, bool)  # step text, instruction, frame ok

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = "RECOGNITION" # "RECOGNITION", "CAPTURE", "IDLE"
        self.mutex = QMutex()
        self.guide = None
        self.capture_dir = ""
        self.captured_files = []
        self._last_guidance = None
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

        # Camera Setup (USB camera)
        cap = open_usb_camera()
        if not cap.isOpened():
            return

        last_name = None
        consecutive = 0
        frame_count = 0

        while self._run_flag:
            current_mode = self.get_mode()
            frame_count += 1

            ret, cv_img = cap.read()
            if not ret:
                self.msleep(40)
                continue

            # Recognition runs only during a START scan, so check every frame for a fast result
            if current_mode == "RECOGNITION":
                self.process_recognition(cv_img, last_name, consecutive)
            elif current_mode == "CAPTURE":
                # Capture mode needs higher FPS for smooth UI feedback
                self.process_capture(cv_img)
            
            # Convert to Qt
            # Fix Color Issue: Ensure input is treated as BGR and converted to RGB
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            # Copy is CRITICAL for thread safety with numpy data
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)
            
            # Prevent CPU starvation; shorter pause while scanning so results come quickly
            self.msleep(10 if current_mode == "RECOGNITION" else 40)

        # Cleanup
        cap.release()

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

        # One status per processed frame so the UI can count consecutive matches
        known = [n for n in names if n != "Unknown"]
        if known:
            self.attendance_signal.emit(f"MATCH:{known[0]}")
        elif names:
            self.attendance_signal.emit("UNKNOWN")
        else:
            self.attendance_signal.emit("NOFACE")

    def process_capture(self, img):
        guide = self.guide
        if guide is None or self.recognizer is None or self.recognizer.detector is None:
            return

        try:
            if not self.capture_dir or not os.path.exists(self.capture_dir):
                print(f"Error: Capture directory missing: {self.capture_dir}")
                self.set_mode("IDLE")
                return

            clean = img.copy()  # saved samples must not contain the overlay
            h, w, _ = img.shape
            self.recognizer.detector.setInputSize((w, h))
            _, faces = self.recognizer.detector.detect(clean)

            ok, message, face = guide.evaluate(clean, faces)

            if ok and guide.ready_for_sample():
                crop = crop_face(clean, face)
                if crop.size > 0:
                    filename = os.path.join(
                        self.capture_dir, f"{guide.stage_key.lower()}_{int(time.time() * 1000)}.jpg")
                    cv2.imwrite(filename, crop)  # USB camera frames are already BGR
                    self.captured_files.append(filename)
                    guide.accept_sample()
                    self.capture_progress_signal.emit(guide.progress)

                    if guide.done:
                        self.set_mode("IDLE")
                        self._emit_guidance("Done", "Face data captured - processing...", True)
                        self.attendance_signal.emit("CAPTURE_COMPLETE")
                        return
                    if guide.stage_count == 0:  # just moved to the next pose
                        ok, message = False, guide.instruction

            self._draw_guide_overlay(img, face, ok, message)
            self._emit_guidance(guide.stage_text, message, ok)
        except Exception as e:
            print(f"Capture Error: {e}")
            self.set_mode("IDLE") # Reset to safe state

    def _emit_guidance(self, step, message, ok):
        # Only emit on change to avoid flooding the UI thread
        state = (step, message, ok)
        if state != self._last_guidance:
            self._last_guidance = state
            self.guidance_signal.emit(step, message, ok)

    def _draw_guide_overlay(self, img, face, ok, message):
        h, w, _ = img.shape
        color = (0, 200, 0) if ok else (0, 165, 255)  # green / orange (BGR)
        # Target oval where the face should be
        cv2.ellipse(img, (w // 2, h // 2), (int(w * 0.20), int(h * 0.36)), 0, 0, 360, color, 3)
        if face is not None:
            x, y, bw, bh = face[:4].astype(int)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        # Instruction banner
        cv2.rectangle(img, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(img, message, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def start_capture(self, user_id, user_name):
        self.capture_dir = os.path.join(KNOWN_FACES_DIR, f"{user_id}_{user_name}")
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
        self.guide = RegistrationGuide()
        self.captured_files = []
        self._last_guidance = None
        self.set_mode("CAPTURE")

    def cancel_capture(self):
        """Stop capturing and remove the samples saved in this session."""
        self.set_mode("IDLE")
        for f in self.captured_files:
            try:
                os.remove(f)
            except OSError:
                pass
        self.captured_files = []
        if self.capture_dir and os.path.isdir(self.capture_dir) and not os.listdir(self.capture_dir):
            os.rmdir(self.capture_dir)

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
                # Dashboard may send: [...] or {"users": [...]}
                if isinstance(payload, dict) and "users" in payload:
                    payload = payload["users"]   # unwrap
                elif isinstance(payload, dict):
                    payload = [payload]           # single user dict
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

def create_machine_output(kind):
    """'relay' -> GPIO relay, 'beckhoff' -> Beckhoff controller over ADS."""
    if kind == "beckhoff":
        from device.beckhoff import BeckhoffOutput   # needs pyads only in this mode
        return BeckhoffOutput()
    return MachineRelay()


# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self, output=MACHINE_OUTPUT):
        super().__init__()
        self.setWindowTitle("Autonex SmartAccess")
        self.setStyleSheet(STYLE_MAIN)

        # Home scan state
        self.scan_state = "IDLE"      # IDLE | SCANNING | RESULT
        self.scan_started = 0.0
        self.match_identity = None
        self.match_count = 0
        self.face_seen = False
        self.machine_on = False       # relay ON after ACCESS GRANTED, OFF after STOP
        self.operator = ("", "")      # (name, id) of the worker who started the machine
        
        self.db = LocalDatabase()
        # Machine enable output (LED for now), OFF at start: GPIO relay or Beckhoff controller
        self.machine_out = create_machine_output(output)
        
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

        # On-screen keyboard: opens when any text box gets focus
        self.keyboard = VirtualKeyboard(self)
        QApplication.instance().installEventFilter(self)
        
        # NOW start the video thread after all widgets exist
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_video_feed)
        self.thread.attendance_signal.connect(self.handle_video_signal)
        self.thread.capture_progress_signal.connect(self.update_capture_progress)
        self.thread.guidance_signal.connect(self.update_guidance)
        self.thread.start()

        self.train_thread = TrainThread()
        self.train_thread.finished_signal.connect(self.on_training_complete)

        # MQTT worker — listens on receive-users and auto-refreshes employee list
        self.mqtt_worker = MQTTWorker()
        self.mqtt_worker.users_updated.connect(self.refresh_employee_list)
        self.mqtt_worker.start()

        self.reg_identity = None
        self.reset_home()

    def init_home_screen(self):
        """Main kiosk screen: live camera on one half, START / result panel on the other."""
        self.home_widget = QWidget()
        # Side by side on a landscape panel, stacked on a portrait one (see resizeEvent)
        self.home_layout = QBoxLayout(QBoxLayout.LeftToRight, self.home_widget)
        self.apply_orientation()
        self.home_layout.setContentsMargins(16, 16, 16, 16)
        self.home_layout.setSpacing(16)

        # ── Camera panel ──────────────────────────────────────────────────
        cam_panel = QFrame()
        cam_panel.setObjectName("camPanel")
        cam_panel.setStyleSheet(f"#camPanel {{ background-color: #000; border: 2px solid {C_RAISED}; border-radius: 16px; }}")
        cam_layout = QGridLayout(cam_panel)
        cam_layout.setContentsMargins(4, 4, 4, 4)

        self.video_container = QLabel("Starting camera...")
        self.video_container.setAlignment(Qt.AlignCenter)
        self.video_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_container.setStyleSheet("color: #6c7086; font-size: 20px; background: transparent;")
        cam_layout.addWidget(self.video_container, 0, 0)

        self.lbl_cam_badge = QLabel("●  LIVE")
        self.lbl_cam_badge.setStyleSheet(
            f"background-color: rgba(17,17,27,200); color: {C_GREEN}; font-size: 16px; font-weight: bold;"
            "padding: 6px 14px; border-radius: 14px; margin: 12px;")
        cam_layout.addWidget(self.lbl_cam_badge, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        self.lbl_cam_hint = QLabel("Stand in front of the camera")
        self.lbl_cam_hint.setStyleSheet(
            f"background-color: rgba(17,17,27,210); color: {C_TEXT}; font-size: 20px; font-weight: bold;"
            "padding: 10px 22px; border-radius: 18px; margin: 16px;")
        cam_layout.addWidget(self.lbl_cam_hint, 0, 0, Qt.AlignBottom | Qt.AlignHCenter)

        # ── Control panel ─────────────────────────────────────────────────
        panel = QFrame()
        panel.setObjectName("ctrlPanel")
        panel.setStyleSheet(f"#ctrlPanel {{ background-color: {C_PANEL}; border: 1px solid {C_RAISED}; border-radius: 16px; }}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(28, 18, 28, 18)
        v.setSpacing(6)

        # Header: brand + network status
        header = QHBoxLayout()
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        lbl_brand = QLabel("Autonex SmartAccess")
        lbl_brand.setStyleSheet(f"color: {C_BLUE}; font-size: 26px; font-weight: bold; letter-spacing: 1px;")
        lbl_sub = QLabel(f"Machine Access  •  Device {DEVICE_ID}")
        lbl_sub.setStyleSheet(f"color: {C_MUTED}; font-size: 15px;")
        brand_box.addWidget(lbl_brand)
        brand_box.addWidget(lbl_sub)
        header.addLayout(brand_box)
        header.addStretch()
        v.addLayout(header)

        # Clock
        self.lbl_clock = QLabel("00:00:00")
        self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet(f"color: {C_TEXT}; font-size: 56px; font-weight: bold;")
        self.lbl_date = QLabel("")
        self.lbl_date.setAlignment(Qt.AlignCenter)
        self.lbl_date.setStyleSheet(f"color: {C_MUTED}; font-size: 20px;")
        v.addWidget(self.lbl_clock)
        v.addWidget(self.lbl_date)

        # Action area: START button (page 0) / result card (page 1)
        self.home_stack = QStackedWidget()

        page_btn = QWidget()
        pb = QVBoxLayout(page_btn)
        pb.setContentsMargins(0, 0, 0, 0)
        self.btn_scan = ScanButton()
        self.btn_scan.clicked.connect(self.on_main_button)
        pb.addWidget(self.btn_scan, alignment=Qt.AlignCenter)
        self.home_stack.addWidget(page_btn)

        page_result = QFrame()
        page_result.setObjectName("resultCard")
        rc = QVBoxLayout(page_result)
        rc.setContentsMargins(20, 10, 20, 10)
        rc.setSpacing(8)
        rc.addStretch()
        self.lbl_result_icon = QLabel("✓")
        self.lbl_result_icon.setFixedSize(120, 120)
        self.lbl_result_icon.setAlignment(Qt.AlignCenter)
        self.lbl_result_title = QLabel("")
        self.lbl_result_title.setAlignment(Qt.AlignCenter)
        self.lbl_result_name = QLabel("")
        self.lbl_result_name.setAlignment(Qt.AlignCenter)
        self.lbl_result_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_result_detail = QLabel("")
        self.lbl_result_detail.setAlignment(Qt.AlignCenter)
        self.lbl_result_detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_result_detail.setStyleSheet(f"color: {C_MUTED}; font-size: 20px;")
        self.lbl_result_pill = QLabel("")
        self.lbl_result_pill.setAlignment(Qt.AlignCenter)
        rc.addWidget(self.lbl_result_icon, alignment=Qt.AlignCenter)
        rc.addWidget(self.lbl_result_title)
        rc.addWidget(self.lbl_result_name)
        rc.addWidget(self.lbl_result_detail)
        rc.addWidget(self.lbl_result_pill, alignment=Qt.AlignCenter)
        rc.addStretch()
        self.home_stack.addWidget(page_result)

        v.addWidget(self.home_stack, stretch=1)

        # Live status line
        self.lbl_home_status = QLabel("")
        self.lbl_home_status.setAlignment(Qt.AlignCenter)
        self.lbl_home_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.lbl_home_status.setFixedHeight(56)
        v.addWidget(self.lbl_home_status)

        # Footer: cancel (while scanning) + menu
        footer = QHBoxLayout()
        footer.setSpacing(12)
        self.btn_cancel_scan = QPushButton("CANCEL")
        self.btn_cancel_scan.setFixedHeight(64)
        self.btn_cancel_scan.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {C_RED}; border: 2px solid {C_RED}; font-size: 20px; }}"
            f"QPushButton:pressed {{ background-color: rgba(243,139,168,40); }}")
        self.btn_cancel_scan.clicked.connect(self.reset_home)
        self.btn_menu = QPushButton("MENU")
        self.btn_menu.setFixedSize(180, 64)
        self.btn_menu.setStyleSheet(
            f"QPushButton {{ background-color: {C_RAISED}; color: {C_TEXT}; font-size: 20px; }}"
            f"QPushButton:pressed {{ background-color: {C_BORDER}; }}"
            f"QPushButton:disabled {{ color: #6c7086; }}")
        self.btn_menu.clicked.connect(lambda: self.switch_screen(1))
        self.lbl_machine = QLabel("")
        self.lbl_machine.setFixedHeight(64)
        self.lbl_machine.setAlignment(Qt.AlignCenter)
        footer.addWidget(self.lbl_machine)
        footer.addWidget(self.btn_cancel_scan, stretch=1)
        footer.addStretch()
        footer.addWidget(self.btn_menu)
        v.addLayout(footer)

        self.home_layout.addWidget(cam_panel, stretch=1)
        self.home_layout.addWidget(panel, stretch=1)

        # Timers: clock/network (1 s), scan progress (100 ms), result hold (single shot)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_home_ui)
        self.timer.start(1000)

        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.on_scan_tick)

        self.result_timer = QTimer(self)
        self.result_timer.setSingleShot(True)
        self.result_timer.timeout.connect(self.reset_home)

        self.update_home_ui()
        self.central_widget.addWidget(self.home_widget)

    # --- HOME: SCAN FLOW ---
    def set_home_status(self, text, color=C_MUTED):
        self.lbl_home_status.setText(text)
        self.lbl_home_status.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")

    def reset_home(self):
        """Back to the idle START screen."""
        self.scan_state = "IDLE"
        self.scan_timer.stop()
        self.result_timer.stop()
        if self.central_widget.currentIndex() == 0 and hasattr(self, "thread"):
            self.thread.set_mode("IDLE")   # live preview only, no recognition
        self.home_stack.setCurrentIndex(0)
        self.lbl_clock.show()
        self.lbl_date.show()
        self.btn_cancel_scan.hide()
        self.btn_menu.setEnabled(True)
        self.update_machine_pill()
        if self.machine_on:
            name, uid = self.operator
            self.btn_scan.set_stop()
            self.lbl_cam_hint.setText("Machine running")
            self.set_home_status(f"Machine ON  —  {name}  (ID {uid})", C_GREEN)
        else:
            self.btn_scan.set_ready()
            self.lbl_cam_hint.setText("Stand in front of the camera")
            self.set_home_status("Press START to turn ON the machine")

    def update_machine_pill(self):
        on = self.machine_on
        color = C_GREEN if on else C_MUTED
        self.lbl_machine.setText("●  MACHINE ON" if on else "●  MACHINE OFF")
        self.lbl_machine.setStyleSheet(
            f"color: {color}; background-color: {C_CARD}; border: 2px solid {color if on else C_RAISED};"
            "border-radius: 14px; padding: 0 18px; font-size: 18px; font-weight: bold;")
        self.lbl_machine.setVisible(self.scan_state != "SCANNING")

    def on_main_button(self):
        if self.machine_on:
            self.stop_machine()
        else:
            self.start_scan()

    def stop_machine(self):
        """STOP pressed: switch the relay off immediately (no face scan needed)."""
        if self.machine_out.off() is False:
            # Controller unreachable: the machine may still be running, keep STOP available
            self.set_home_status("Could not stop machine  —  controller not reachable", C_RED)
            return
        self.machine_on = False
        self.operator = ("", "")
        self.reset_home()
        self.set_home_status("Machine stopped  —  press START to turn it ON", C_YELLOW)

    def start_scan(self):
        if self.scan_state != "IDLE":
            return
        self.scan_state = "SCANNING"
        self.scan_started = time.time()
        self.match_identity, self.match_count, self.face_seen = None, 0, False
        self.btn_scan.set_scanning(0)
        self.lbl_machine.hide()
        self.btn_cancel_scan.show()
        self.btn_menu.setEnabled(False)
        self.lbl_cam_hint.setText("Look directly at the camera")
        self.set_home_status("Scanning face...", C_BLUE)
        self.scan_timer.start(100)
        self.thread.set_mode("RECOGNITION")

    def on_scan_tick(self):
        elapsed = time.time() - self.scan_started
        self.btn_scan.set_scanning(elapsed / SCAN_TIMEOUT_S)
        if elapsed >= SCAN_TIMEOUT_S:
            if self.face_seen:
                self.show_result("FAIL", "ACCESS DENIED", "Face not recognised",
                                 "Try again or contact the administrator")
            else:
                self.show_result("FAIL", "NO FACE DETECTED", "",
                                 "Stand in front of the camera")

    def handle_home_recognition(self, msg):
        if self.scan_state != "SCANNING":
            return
        # An unclear frame does not reset the count; only a different person does
        if msg == "NOFACE":
            self.lbl_cam_hint.setText("No face  —  look at the camera")
            if not self.match_count:
                self.set_home_status("Looking for a face...", C_BLUE)
        elif msg == "UNKNOWN":
            self.face_seen = True
            self.lbl_cam_hint.setText("Hold still")
            if not self.match_count:
                self.set_home_status("Verifying...", C_BLUE)
        elif msg.startswith("MATCH:"):
            self.face_seen = True
            identity = msg[len("MATCH:"):]
            if identity == self.match_identity:
                self.match_count += 1
            else:
                self.match_identity, self.match_count = identity, 1
            self.lbl_cam_hint.setText("Hold still")
            self.set_home_status(f"Verifying...  {self.match_count}/{VERIFICATION_FRAMES}", C_BLUE)
            if self.match_count >= VERIFICATION_FRAMES:
                self.complete_scan(identity)

    def complete_scan(self, identity):
        # Folder / identity format is "ID_Name" (or just "Name" for old data)
        user_id, name = identity.split("_", 1) if "_" in identity else (identity, identity)
        self.db.add_record(DEVICE_ID, name, user_id=user_id)
        if self.machine_out.grant() is False:   # enable the machine
            self.show_result("FAIL", "MACHINE NOT STARTED", name,
                             "Controller not reachable  —  check the connection")
            return
        self.machine_on = True
        self.operator = (name, user_id)
        self.show_result("OK", "ACCESS GRANTED", name, f"ID {user_id}   •   Machine ON")

    def show_result(self, kind, title, name, detail, pill=""):
        self.scan_state = "RESULT"
        self.scan_timer.stop()
        self.thread.set_mode("IDLE")
        color, icon = {"OK": (C_GREEN, "✓"), "FAIL": (C_RED, "✕")}[kind]
        self.lbl_result_icon.setText(icon)
        self.lbl_result_icon.setStyleSheet(
            f"background-color: {color}; color: {C_BG}; border-radius: 60px; font-size: 64px; font-weight: bold;")
        self.lbl_result_title.setText(title)
        self.lbl_result_title.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold; letter-spacing: 2px;")
        self.lbl_result_name.setText(name)
        # Shrink long names instead of letting them widen the panel
        name_px = 36 if len(name) <= 18 else 28 if len(name) <= 26 else 22
        self.lbl_result_name.setStyleSheet(f"color: {C_TEXT}; font-size: {name_px}px; font-weight: bold;")
        self.lbl_result_name.setVisible(bool(name))
        self.lbl_result_detail.setText(detail)
        self.lbl_result_pill.setText(pill)
        self.lbl_result_pill.setVisible(bool(pill))
        self.lbl_result_pill.setStyleSheet(
            f"color: {color}; border: 2px solid {color}; border-radius: 16px; padding: 6px 18px;"
            "font-size: 18px; font-weight: bold;")
        self.lbl_clock.hide()   # give the result card the full panel height
        self.lbl_date.hide()
        self.home_stack.setCurrentIndex(1)
        self.btn_cancel_scan.hide()
        self.update_machine_pill()
        self.btn_menu.setEnabled(True)
        self.lbl_cam_hint.setText("Thank you" if kind == "OK" else "Press START to try again")
        self.set_home_status("Returning to start...", C_MUTED)
        self.result_timer.start(RESULT_HOLD_MS)

    def init_settings_screen(self):
        self.settings_widget = QWidget()
        main_layout = QVBoxLayout(self.settings_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self.create_top_bar("MENU", lambda: self.switch_screen(0)))

        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(32, 28, 32, 28)
        grid_layout.setSpacing(20)

        self.create_grid_btn(grid_layout, "User Management", "Add, list and remove users", 0, 0,
                             lambda: self.switch_screen(6), C_BLUE)
        self.create_grid_btn(grid_layout, "Shift", "Working hours and grace time", 0, 1,
                             lambda: self.switch_screen(7), C_YELLOW)
        self.create_grid_btn(grid_layout, "Communication", "Device, Ethernet and WiFi", 1, 0,
                             lambda: self.switch_screen(8), C_GREEN)
        self.create_grid_btn(grid_layout, "System Info", "Network address and version", 1, 1,
                             self.show_about_screen, C_MUTED)

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
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left: Form
        form_container = QFrame()
        form_container.setObjectName("regForm")
        form_container.setStyleSheet(f"#regForm {{ background-color: {C_PANEL}; border: 1px solid {C_RAISED}; border-radius: 16px; }}")
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(28, 24, 28, 24)
        form_layout.setSpacing(14)

        lbl_title = QLabel("New User Registration")
        lbl_title.setStyleSheet(f"color: {C_BLUE}; font-size: 28px; font-weight: bold;")

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Full Name")

        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Employee ID")

        self.btn_start = QPushButton("Start Scanning")
        self.btn_start.setFixedHeight(68)
        self.btn_start.setStyleSheet(
            f"QPushButton {{ background-color: {C_GREEN}; color: {C_BG}; font-size: 22px; }}"
            "QPushButton:pressed { background-color: #8fd18a; }")
        self.btn_start.clicked.connect(self.start_registration)

        self.btn_cancel_reg = QPushButton("Cancel")
        self.btn_cancel_reg.setFixedHeight(60)
        self.btn_cancel_reg.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {C_RED}; border: 2px solid {C_RED}; font-size: 20px; }}"
            "QPushButton:pressed { background-color: rgba(243,139,168,40); }")
        self.btn_cancel_reg.clicked.connect(self.cancel_registration)

        self.progress_ring = CircularProgress()
        self.progress_ring.hide()

        # Guided registration: current pose step + live instruction
        self.lbl_step = QLabel("")
        self.lbl_step.setAlignment(Qt.AlignCenter)
        self.lbl_step.setStyleSheet(f"color: {C_BLUE}; font-size: 20px; font-weight: bold;")
        self.lbl_step.hide()

        self.lbl_status = QLabel("Ready to Scan")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {C_TEXT}; font-size: 20px;")

        form_layout.addWidget(lbl_title)
        form_layout.addSpacing(6)
        form_layout.addWidget(self.input_name)
        form_layout.addWidget(self.input_id)
        form_layout.addWidget(self.progress_ring, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.lbl_step)
        form_layout.addWidget(self.lbl_status)
        form_layout.addStretch()
        form_layout.addWidget(self.btn_start)
        form_layout.addWidget(self.btn_cancel_reg)

        # Right: Camera Preview
        self.video_label_reg = QLabel()
        self.video_label_reg.setAlignment(Qt.AlignCenter)
        self.video_label_reg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label_reg.setStyleSheet(f"background-color: #000; border: 2px solid {C_RAISED}; border-radius: 16px;")

        layout.addWidget(form_container, stretch=2)
        layout.addWidget(self.video_label_reg, stretch=3)

        self.central_widget.addWidget(self.reg_widget)

    def init_delete_screen(self):
        self.del_widget = QWidget()
        main_layout = QVBoxLayout(self.del_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.create_top_bar("Delete User", lambda: self.switch_screen(6)))
        
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
        
        main_layout.addWidget(self.create_top_bar("System Info", lambda: self.switch_screen(1)))
        
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
        now = datetime.now()
        self.lbl_clock.setText(now.strftime("%H:%M:%S"))
        self.lbl_date.setText(now.strftime("%A, %d %B %Y"))
        if self.machine_on and not self.machine_out.is_on:
            self.machine_on = False
            if self.scan_state == "IDLE":
                self.reset_home()

    def switch_screen(self, index):
        if hasattr(self, "keyboard"):
            self.keyboard.close_keyboard()
        self.central_widget.setCurrentIndex(index)
        if index == 0:
            self.reset_home()
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
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top Bar
        top_bar = self.create_top_bar("User Management", lambda: self.switch_screen(1))
        layout.addWidget(top_bar)
        
        # Grid
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setContentsMargins(32, 28, 32, 28)
        grid.setSpacing(20)
        
        self.create_grid_btn(grid, "Add User",      "Register a new face", 0, 0, lambda: self.switch_screen(2), C_GREEN)
        self.create_grid_btn(grid, "Employee List", "Synced from dashboard", 0, 1, lambda: self.switch_screen(12), C_BLUE)
        self.create_grid_btn(grid, "User View",     "Registered faces on device", 1, 0, self.refresh_user_view_and_show, C_MUTED)
        self.create_grid_btn(grid, "Delete User",   "Remove a registered face", 1, 1, self.refresh_delete_list_and_show, C_RED)
        
        layout.addWidget(grid_container)
        self.central_widget.addWidget(self.user_mgt_widget)

    def init_user_view_screen(self):
        self.user_view_widget = QWidget()
        layout = QVBoxLayout(self.user_view_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
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
        layout.setContentsMargins(0, 0, 0, 0)
        
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
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.create_top_bar("Communication", lambda: self.switch_screen(1)))
        
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setContentsMargins(32, 28, 32, 28)
        grid.setSpacing(20)
        
        self.create_grid_btn(grid, "Comm Params", "Device ID and port", 0, 0, lambda: self.switch_screen(9), C_BLUE)
        self.create_grid_btn(grid, "Ethernet", "Wired network settings", 0, 1, lambda: self.switch_screen(10), C_GREEN)
        self.create_grid_btn(grid, "WiFi", "Wireless network", 1, 0, lambda: self.switch_screen(11), C_YELLOW)
        
        layout.addWidget(grid_container)
        self.central_widget.addWidget(self.comm_menu_widget)

    def init_comm_params_screen(self):
        self.comm_params_widget = QWidget()
        layout = QVBoxLayout(self.comm_params_widget)
        layout.setContentsMargins(0, 0, 0, 0)
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
        layout.setContentsMargins(0, 0, 0, 0)
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.create_top_bar("WiFi Settings", lambda: self.switch_screen(8)))
        
        lbl = QLabel("WiFi Scanning not implemented yet.")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        
        self.central_widget.addWidget(self.wifi_widget)

    # --- HELPERS ---
    def create_top_bar(self, title, back_callback):
        frame = QFrame()
        frame.setObjectName("topBar")
        frame.setStyleSheet(f"#topBar {{ background-color: {C_PANEL}; border-bottom: 2px solid {C_RAISED}; }}")
        frame.setFixedHeight(84)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)

        btn = QPushButton("‹  BACK")
        btn.setFixedSize(160, 62)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {C_RAISED}; color: {C_TEXT}; font-size: 20px; }}"
            f"QPushButton:pressed {{ background-color: {C_BORDER}; }}")
        btn.clicked.connect(back_callback)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 28px; font-weight: bold; letter-spacing: 1px;")
        lbl.setAlignment(Qt.AlignCenter)

        layout.addWidget(btn)
        layout.addWidget(lbl, stretch=1)
        # Spacer the same width as the back button keeps the title centred
        d = QWidget(); d.setFixedSize(160, 10); layout.addWidget(d)

        return frame

    def create_grid_btn(self, layout, title, subtitle, row, col, callback, accent=C_BLUE):
        """Large touch tile with a title, a one-line description and a coloured accent edge."""
        btn = QPushButton()
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn.setMinimumHeight(120)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {C_CARD}; border: 1px solid {C_RAISED};"
            f" border-radius: 14px; padding: 0; }}"
            f"QPushButton:pressed {{ background-color: {C_RAISED}; }}")
        row_layout = QHBoxLayout(btn)
        row_layout.setContentsMargins(14, 18, 20, 18)
        row_layout.setSpacing(18)
        stripe = QFrame()
        stripe.setFixedWidth(6)
        stripe.setStyleSheet(f"background-color: {accent}; border: none; border-radius: 3px;")
        stripe.setAttribute(Qt.WA_TransparentForMouseEvents)
        row_layout.addWidget(stripe)
        inner = QVBoxLayout()
        inner.setSpacing(6)
        row_layout.addLayout(inner, stretch=1)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: bold; background: transparent; border: none;")
        lbl_sub = QLabel(subtitle)
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet(f"color: {C_MUTED}; font-size: 17px; background: transparent; border: none;")
        for lbl in (lbl_title, lbl_sub):
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        inner.addStretch()
        inner.addWidget(lbl_title)
        inner.addWidget(lbl_sub)
        inner.addStretch()
        if callback:
            btn.clicked.connect(callback)
        layout.addWidget(btn, row, col)
        return btn

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
            self.lbl_status.setStyleSheet(f"color: {C_RED}; font-size: 20px;")
            return
        
        self.btn_start.hide()
        self.btn_cancel_reg.show()  # allow aborting while scanning
        self.input_name.setEnabled(False)
        self.input_id.setEnabled(False)
        self.progress_ring.set_value(0)
        self.progress_ring.show()
        self.lbl_step.show()
        self.lbl_status.setText("Look directly at the camera")
        self.lbl_status.setStyleSheet(f"color: {C_TEXT}; font-size: 20px;")

        self.reg_identity = f"{uid}_{name}"
        self.thread.start_capture(uid, name)

    def cancel_registration(self):
        if self.thread.get_mode() == "CAPTURE":
            self.thread.cancel_capture()
        self.reset_registration()

    def update_guidance(self, step, message, ok):
        if self.central_widget.currentIndex() != 2:
            return
        self.lbl_step.setText(step)
        self.lbl_status.setText(message)
        color = "#a6e3a1" if ok else "#fab387"  # green when capturing, orange when user must adjust
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")

    def count_trained_samples(self, identity):
        """Number of embeddings stored for this user after training."""
        try:
            with open(NAMES_FILE, 'r') as f:
                return sum(1 for n in json.load(f) if n == identity)
        except Exception:
            return 0

    def update_video_feed(self, img):
        current_idx = self.central_widget.currentIndex()
        if current_idx == 0:
            # Home: mirrored like a selfie view, filled to the panel without stretching
            target, fill = self.video_container, True
            img = img.mirrored(True, False)
        elif current_idx == 2:
            # Registration: whole frame visible (not mirrored, so left/right instructions match)
            target, fill = self.video_label_reg, False
        else:
            return

        try:
            size = target.size()
            pixmap = QPixmap.fromImage(img)
            if size.width() > 10 and size.height() > 10:
                if fill:
                    pixmap = pixmap.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
                    x = (pixmap.width() - size.width()) // 2
                    y = (pixmap.height() - size.height()) // 2
                    pixmap = pixmap.copy(x, y, size.width(), size.height())
                else:
                    pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.FastTransformation)
            target.setPixmap(pixmap)
        except Exception:
            # Silently ignore any Qt errors during screen transitions
            pass

    def handle_video_signal(self, msg):
        current_idx = self.central_widget.currentIndex()
        if current_idx == 0: # Home
            self.handle_home_recognition(msg)
        elif current_idx == 2: # Register
            if msg == "CAPTURE_COMPLETE":
                self.lbl_status.setText("Processing Profile...")
                self.train_thread.start()

    def update_capture_progress(self, val):
        self.progress_ring.set_value(val)

    def on_training_complete(self, success, msg):
        if self.central_widget.currentIndex() == 2: # Register Mode
            if success:
                self.thread.reload_model()
                # Complete only if enough samples produced a usable embedding
                valid = self.count_trained_samples(self.reg_identity)
                if valid >= MIN_VALID_SAMPLES:
                    self.lbl_step.setText("Done")
                    self.lbl_status.setText(f"Registration Complete! ({valid} face samples)")
                    self.lbl_status.setStyleSheet(f"color: {C_GREEN}; font-size: 22px; font-weight: bold;")
                    QTimer.singleShot(2000, self.reset_registration)
                    return
                self.lbl_status.setText(
                    f"Not enough good face data ({valid}/{MIN_VALID_SAMPLES}). "
                    f"Improve lighting and press Start Scanning again.")
            else:
                self.lbl_status.setText("Error: " + msg)
            self.lbl_status.setStyleSheet(f"color: {C_RED}; font-size: 20px;")
            self.btn_start.setText("Scan Again")
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
        self.input_name.setEnabled(True)
        self.input_id.setEnabled(True)
        self.btn_start.setText("Start Scanning")
        self.btn_start.show()
        self.btn_cancel_reg.show()
        self.progress_ring.hide()
        self.lbl_step.hide()
        self.lbl_status.setText("Ready")
        self.lbl_status.setStyleSheet(f"color: {C_TEXT}; font-size: 20px;")
        self.thread.set_mode("IDLE")  # Ensure we stop scanning when resetting

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
        hint = QLabel("Tap a yellow row to register that employee's face")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {C_YELLOW}; font-size: 17px; padding: 8px;")
        layout.addWidget(hint)

        # List widget
        self.emp_list_view = QListWidget()
        self.emp_list_view.setFont(QFont("Segoe UI", 16))
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
        btn_refresh = QPushButton("Refresh List")
        btn_refresh.setFixedHeight(64)
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 0px;
                font-size: 20px;
            }
            QPushButton:pressed { background-color: #45475a; }
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
                badge = "✓"
                color = "#a6e3a1"   # green
                left_border = "#a6e3a1"
            else:
                badge = "!"
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
        summary = QListWidgetItem(f"  {reg_count} of {total} employees registered")
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
        self.lbl_status.setStyleSheet(f"color: {C_TEXT}; font-size: 20px;")
        self.btn_start.show()
        self.btn_cancel_reg.show()
        self.progress_ring.hide()
        self.switch_screen(2)  # Go to Register screen (index 2)

    def closeEvent(self, event):
        self.machine_out.close()
        self.thread.stop()
        self.mqtt_worker.stop()
        self.mqtt_worker.wait()
        event.accept()

    def apply_orientation(self):
        # Landscape panel: camera | controls side by side. Portrait panel: camera on top.
        # Uses the physical screen size (the window itself can't shrink below the landscape layout).
        screen = QApplication.primaryScreen()
        size = screen.size() if screen else self.size()
        portrait = size.height() > size.width()
        self.home_layout.setDirection(QBoxLayout.TopToBottom if portrait else QBoxLayout.LeftToRight)

    def eventFilter(self, obj, event):
        # Show the on-screen keyboard when the user taps a text box in this window
        # (not when Qt moves focus by itself, e.g. on entering a screen)
        etype = event.type()
        if (etype == QEvent.MouseButtonPress and isinstance(obj, QLineEdit)
                and obj.isEnabled() and obj.window() is self):
            self.keyboard.show_for(obj)
        elif etype == QEvent.FocusOut and isinstance(obj, QLineEdit):
            QTimer.singleShot(0, self._hide_keyboard_if_unfocused)
        return False

    def _hide_keyboard_if_unfocused(self):
        if not isinstance(QApplication.focusWidget(), QLineEdit):
            self.keyboard.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "keyboard") and self.keyboard.isVisible():
            self.keyboard.place()
        self.apply_orientation()   # follows a display rotation at runtime

    def keyPressEvent(self, event):
        # Maintenance with a keyboard attached: F11 toggles full screen, Esc leaves it
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(event)


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

    # Machine output: python3 hmi.py --output relay | beckhoff  (default from shared/config.py)
    import argparse
    parser = argparse.ArgumentParser(description="Autonex SmartAccess HMI")
    parser.add_argument("--output", choices=["relay", "beckhoff"], default=MACHINE_OUTPUT,
                        help="machine enable output (default: %(default)s)")
    args, qt_args = parser.parse_known_args()
    print(f"Machine output: {args.output}")

    app = QApplication(sys.argv[:1] + qt_args)
    
    # Font (falls back to the system sans font on Raspberry Pi OS)
    font = QFont("Segoe UI", 12)
    app.setFont(font)

    try:
        window = MainApp(output=args.output)
        window.showFullScreen()  # kiosk: fill the whole 1280x720 panel
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application Crashed: {e}")

