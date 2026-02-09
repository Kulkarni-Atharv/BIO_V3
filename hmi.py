
import sys
import cv2
import os
import time
import numpy as np

# Try to import picamera2 for Raspberry Pi CSI cameras
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QMessageBox, QListWidget)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


# Import modules
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from core.face_encoder import FaceEncoder
from shared.config import DEVICE_ID, KNOWN_FACES_DIR, VERIFICATION_FRAMES


# --- Worker Thread for Video & Recognition ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    attendance_signal = pyqtSignal(str) # Emits name when recognized

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode = "RECOGNITION" # or "CAPTURE"
        self.capture_count = 0
        self.capture_target = 30
        self.capture_dir = ""
        self.capture_id = ""
        
        # Initialize Logic in RUN to avoid thread affinity issues
        self.recognizer = None

    def run(self):
        # Initialize Recognizer here (Worker Thread)
        if self.recognizer is None:
            self.recognizer = FaceRecognizer()

        # Try multiple camera backends
        cap = None
        picam2 = None
        use_picamera2 = False
        
        # Option 1: Try picamera2 for Raspberry Pi CSI camera
        if PICAMERA2_AVAILABLE:
            try:
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                picam2.configure(config)
                picam2.start()
                # Set controls to ensure color
                picam2.set_controls({"AeEnable": True, "AwbEnable": True, "Saturation": 1.0, "AwbMode": 1}) # 1=Auto
                use_picamera2 = True
                print("Using picamera2 for CSI camera (Color Mode Enabled)")
            except Exception as e:
                print(f"picamera2 failed: {e}")
                use_picamera2 = False
        
        # Option 2: Try V4L2 backend (USB cameras on Linux)
        if not use_picamera2:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Option 3: Try default backend
        if not use_picamera2 and (not cap or not cap.isOpened()):
            cap = cv2.VideoCapture(0)
        
        if not use_picamera2 and (not cap or not cap.isOpened()):
            print("ERROR: Could not open any camera")
            return
        
        last_attendance_time = {}
        last_recognized_name = None
        consecutive_frames = 0
        COOLDOWN = 10 # seconds

        try:
            while self._run_flag:
                # Get frame from appropriate source
                if use_picamera2:
                    cv_img = picam2.capture_array()
                    # picamera2 with RGB888 gives RGB, convert to BGR for OpenCV
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    ret = True
                else:
                    ret, cv_img = cap.read()
                
                if ret:
                    # Logic based on mode
                    if self.mode == "RECOGNITION":
                        if self.recognizer:
                            locations, names = self.recognizer.recognize_faces(cv_img)
                            # Draw and Emit
                            # Recognizer now returns (x, y, w, h)
                            for (x, y, w, h), name in zip(locations, names):
                                left, top, right, bottom = x, y, x+w, y+h
                                
                                if name != "Unknown":
                                    # Multi-frame verification
                                    if name == last_recognized_name:
                                        consecutive_frames += 1
                                    else:
                                        last_recognized_name = name
                                        consecutive_frames = 1
                                    
                                    # Visual feedback (Yellow for verifying, Green for verified)
                                    color = (0, 255, 255) # Yellow
                                    if consecutive_frames >= VERIFICATION_FRAMES:
                                        color = (0, 255, 0) # Green
                                        
                                        now = time.time()
                                        if name not in last_attendance_time or (now - last_attendance_time.get(name, 0) > COOLDOWN):
                                            self.attendance_signal.emit(name)
                                            last_attendance_time[name] = now
                                    else:
                                        pass # Just waiting for more frames

                                    cv2.rectangle(cv_img, (left, top), (right, bottom), color, 2)
                                    status_text = f"{name}"
                                    if consecutive_frames < VERIFICATION_FRAMES:
                                        status_text += f" ({consecutive_frames}/{VERIFICATION_FRAMES})"
                                    cv2.putText(cv_img, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                else:
                                     # Unknown face
                                     cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 0, 255), 2)
                                     cv2.putText(cv_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Reset counter if no faces found
                        if not names and last_recognized_name is not None:
                             last_recognized_name = None
                             consecutive_frames = 0
                    
                    elif self.mode == "CAPTURE":
                        # Just detect to show user face is found
                        if self.recognizer and self.recognizer.detector:
                            h, w, _ = cv_img.shape
                            self.recognizer.detector.setInputSize((w, h))
                            _, faces = self.recognizer.detector.detect(cv_img)
                            
                            if faces is not None:
                                for face in faces:
                                    box = face[:4].astype(int)
                                    x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                                    cv2.rectangle(cv_img, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
                                    
                                    # Capture logic
                                    if self.capture_count < self.capture_target:
                                        self.capture_count += 1
                                        # Save
                                        filename = f"{self.capture_dir}/User.{self.capture_id}.{self.capture_count}.jpg"
                                        # Ensure bounds
                                        x = max(0, x); y = max(0, y)
                                        if w_box > 0 and h_box > 0:
                                            crop = cv_img[y:y+h_box, x:x+w_box]
                                            cv2.imwrite(filename, crop)
                                        
                        cv2.putText(cv_img, f"Captured: {self.capture_count}/{self.capture_target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        if self.capture_count >= self.capture_target:
                            self.mode = "IDLE" # Stop capturing
                            self.attendance_signal.emit("CAPTURE_COMPLETE")

                    # Convert to Qt Image
                    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                    self.change_pixmap_signal.emit(p)
        finally:
            if use_picamera2 and picam2:
                picam2.stop()
            elif cap:
                cap.release()

    def start_capture(self, user_id, user_name):
        self.capture_id = user_id
        self.capture_dir = os.path.join(KNOWN_FACES_DIR, f"{user_id}_{user_name}")
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)
        
        self.capture_count = 0
        self.mode = "CAPTURE"

    def stop(self):
        self._run_flag = False
        self.wait()

    def reload_model(self):
        # We need to signal the thread to reload, safely.
        # Simplest way: set flag, let run loop handle it or just re-init next frame?
        # Actually simplest is to just re-instantiate in the run loop if a flag is set.
        # But for now, we can just replace the object (atomic assignment in Python is generally safe for this usage)
        if self.isRunning():
            self.recognizer = FaceRecognizer()
        else:
            self.recognizer = FaceRecognizer()

# ... (MainApp class remains mostly same, just skipped for brevity unless changes needed) ...
# Actually we need to patch the __main__ block


class TrainThread(QThread):
    finished_signal = pyqtSignal(bool, str)

    def run(self):
        try:
            encoder = FaceEncoder()
            success = encoder.process_images()
            if success:
                self.finished_signal.emit(True, "Training Complete")
            else:
                self.finished_signal.emit(False, "No embeddings generated")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Attendance System (HMI)")
        self.setGeometry(100, 100, 1000, 600)
        
        # Database
        self.db = LocalDatabase()

        # Stacked Widget for Screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 1. Home Screen
        self.home_widget = QWidget()
        self.init_home_ui()
        self.stacked_widget.addWidget(self.home_widget)

        # 2. Add User Screen
        self.add_user_widget = QWidget()
        self.init_add_user_ui()
        self.stacked_widget.addWidget(self.add_user_widget)

        # Workers
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.attendance_signal.connect(self.handle_attendance_signal)
        self.thread.start()
        
        self.train_thread = TrainThread()
        self.train_thread.finished_signal.connect(self.handle_training_finished)

    def init_home_ui(self):
        layout = QHBoxLayout()
        # Left: Video
        self.video_label = QLabel("Loading Camera...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        # Right: Sidebar
        sidebar = QVBoxLayout()
        title = QLabel("Attendance Log")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        sidebar.addWidget(title)
        self.log_list = QListWidget()
        sidebar.addWidget(self.log_list)

        btn_add = QPushButton("Add New User")
        btn_add.setFixedHeight(50)
        btn_add.clicked.connect(lambda: self.switch_screen(1))
        sidebar.addWidget(btn_add)

        btn_delete = QPushButton("Delete User")
        btn_delete.setFixedHeight(50)
        btn_delete.setStyleSheet("background-color: #ffcccc;")
        btn_delete.clicked.connect(self.delete_user_action)
        sidebar.addWidget(btn_delete)

        layout.addLayout(sidebar)
        self.home_widget.setLayout(layout)

    def delete_user_action(self):
        # List users from known_faces directory
        known_faces_dir = KNOWN_FACES_DIR
        if not os.path.exists(known_faces_dir):
            QMessageBox.warning(self, "Error", "No known_faces directory found.")
            return

        users = [d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d))]
        if not users:
            QMessageBox.information(self, "Info", "No users found to delete.")
            return

        # Show Selection Dialog
        from PyQt5.QtWidgets import QInputDialog
        user, ok = QInputDialog.getItem(self, "Delete User", "Select user to delete:", users, 0, False)
        
        if ok and user:
            confirm = QMessageBox.question(self, "Confirm Delete", 
                                         f"Are you sure you want to delete '{user}'?\nThis cannot be undone.",
                                         QMessageBox.Yes | QMessageBox.No)
            
            if confirm == QMessageBox.Yes:
                import shutil
                try:
                    shutil.rmtree(os.path.join(known_faces_dir, user))
                    QMessageBox.information(self, "Success", f"User '{user}' deleted.")
                    
                    # Retrain Model
                    self.lbl_status_home = QLabel("Updating Model...") # Quick feedback hack, better to use status bar or just wait
                    # Actually we don't have a status bar on home, let's just use the modal approach or run background
                    # Since we have TrainThread, let's use it.
                    self.train_thread.start()
                    QMessageBox.information(self, "Updating", "Model is updating in background...")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to delete user: {e}")

    def init_add_user_ui(self):
        layout = QVBoxLayout()
        header = QLabel("Register New User")
        header.setFont(QFont("Arial", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        form_layout = QHBoxLayout()
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Enter Name (e.g., John)")
        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Enter ID (numeric)")
        form_layout.addWidget(self.input_name)
        form_layout.addWidget(self.input_id)
        layout.addLayout(form_layout)
        self.video_label_reg = QLabel("Camera Preview")
        self.video_label_reg.setFixedSize(640, 480)
        self.video_label_reg.setStyleSheet("background-color: black;")
        self.video_label_reg.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label_reg, alignment=Qt.AlignCenter)
        btn_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("Start Capture (30 Images)")
        self.btn_capture.clicked.connect(self.start_capture)
        
        # Training is now automatic, but we can keep a hidden or disabled button just in case, 
        # or remove it. User asked for automatic. Let's show specific status button instead.
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)

        btn_back = QPushButton("Back to Home")
        btn_back.clicked.connect(lambda: self.switch_screen(0))
        
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.lbl_status)
        btn_layout.addWidget(btn_back)
        layout.addLayout(btn_layout)
        self.add_user_widget.setLayout(layout)

    def switch_screen(self, index):
        self.stacked_widget.setCurrentIndex(index)
        if index == 0:
            self.thread.mode = "RECOGNITION"
        elif index == 1:
            pass

    def update_image(self, qt_img):
        if self.stacked_widget.currentIndex() == 0:
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        else:
            self.video_label_reg.setPixmap(QPixmap.fromImage(qt_img))

    def handle_attendance_signal(self, payload):
        if payload == "CAPTURE_COMPLETE":
            self.btn_capture.setText("Capture Done! Processing...")
            self.btn_capture.setEnabled(False)
            self.lbl_status.setText("Training Model... Please Wait.")
            
            # Start Training Automatically
            self.train_thread.start()
            return
            
        name = payload
        self.db.add_record(DEVICE_ID, name)
        time_str = time.strftime("%H:%M:%S")
        self.log_list.insertItem(0, f"[{time_str}] {name}")

    def start_capture(self):
        name = self.input_name.text().strip()
        uid = self.input_id.text().strip()
        if not name or not uid:
            QMessageBox.warning(self, "Input Error", "Please enter ID and Name.")
            return
        self.thread.start_capture(uid, name)
        self.btn_capture.setText("Capturing... Look at Camera")
        self.btn_capture.setEnabled(False)
        self.lbl_status.setText("Capturing...")

    def handle_training_finished(self, success, message):
        if success:
            # Check if this was triggered by delete or add
            # Ideally distinguish, but generic message is fine
            if self.stacked_widget.currentIndex() == 0:
               # Home screen (Delete action likely)
               QMessageBox.information(self, "Success", "Model Updated Successfully.")
               self.thread.reload_model()
            else:
               # Add screen
               QMessageBox.information(self, "Success", f"User Registered!\n{message}")
               self.thread.reload_model()
               
               # Reset UI
               self.btn_capture.setText("Start Capture (30 Images)")
               self.btn_capture.setEnabled(True)
               self.lbl_status.setText("Ready (Model Updated)")
               self.input_name.clear()
               self.input_id.clear()
        else:
            QMessageBox.warning(self, "Error", f"Training Failed: {message}")
            if self.stacked_widget.currentIndex() == 1:
                self.lbl_status.setText("Error")
                self.btn_capture.setEnabled(True)
                self.btn_capture.setText("Retry Capture")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    # Fix for OpenCV vs PyQt5 conflict
    # Force usage of PyQt5 plugins inside the venv
    import PyQt5
    plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
