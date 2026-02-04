
import cv2
import os
import json
import logging

logger = logging.getLogger("Recognizer")

class FaceRecognizer:
    def __init__(self, known_faces_dir='known_faces'):
        self.known_faces_dir = known_faces_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}
        self.is_trained = False
        
        self._load_model()

    def _load_model(self):
        if os.path.exists('trainer.yml') and os.path.exists('names.json'):
            try:
                self.recognizer.read('trainer.yml')
                with open('names.json', 'r') as f:
                    # Convert keys back to integers
                    data = json.load(f)
                    self.names = {int(k): v for k, v in data.items()}
                self.is_trained = True
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning("No trainer.yml found. Recognition will be limited to detection only.")

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_locations = []
        face_names = []

        for (x, y, w, h) in faces:
            # Format: top, right, bottom, left
            top, right, bottom, left = y, x + w, y + h, x
            
            name = "Unknown"
            if self.is_trained:
                try:
                    # Predict returns (id, confidence)
                    # Confidence: 0 is perfect match. < 100 is usually good.
                    id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
                    
                    # You may need to tune this threshold
                    if confidence < 100:
                        name = self.names.get(id, f"ID:{id}")
                        # name = f"{name} ({round(confidence)})" # Debug show confidence
                    else:
                        name = "Unknown"
                except Exception as e:
                    logger.error(f"Prediction error: {e}")

            face_locations.append((top, right, bottom, left))
            face_names.append(name)

        return face_locations, face_names
