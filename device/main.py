
import cv2
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from device.camera import Camera
from core.recognizer import FaceRecognizer
from device.database import LocalDatabase
from device.uploader import DataUploader
from shared.config import DEVICE_ID, KNOWN_FACES_DIR

def main():
    print("Starting Smart Attendance System Device...")

    # Initialize modules
    db = LocalDatabase()
    uploader = DataUploader(db)
    recognizer = FaceRecognizer(known_faces_dir=KNOWN_FACES_DIR)
    camera = Camera(source=0) # Using default webcam

    # Start independent threads
    uploader.start()
    camera.start()

    # Cooldown mechanism to avoid spamming the same attendance
    last_attendance = {}
    COOLDOWN_SECONDS = 60

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                time.sleep(0.1)
                continue

            # Perform recognition
            face_locations, face_names = recognizer.recognize_faces(frame)

            # Process results
            current_time = time.time()
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                if name != "Unknown":
                    # Check cooldown
                    if name not in last_attendance or (current_time - last_attendance[name] > COOLDOWN_SECONDS):
                        print(f"Marking attendance for: {name}")
                        db.add_record(DEVICE_ID, name)
                        last_attendance[name] = current_time

            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.stop()
        uploader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
