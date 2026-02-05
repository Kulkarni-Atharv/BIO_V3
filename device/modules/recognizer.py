import cv2
import os
import json
import logging
import numpy as np

logger = logging.getLogger("Recognizer")

class FaceRecognizer:
    def __init__(self, known_faces_dir='known_faces'):
        self.known_faces_dir = known_faces_dir
        self.yunet_path = 'face_detection_yunet_2023mar.onnx'
        self.mobilefacenet_path = 'MobileFaceNet.onnx'
        self.embeddings_file = 'embeddings.npy'
        self.names_file = 'names.json'
        
        self.detector = None
        self.recognizer = None
        self.known_embeddings = []
        self.known_names = []
        self.is_trained = False
        
        self._load_models()
        self._load_embeddings()

    def _load_models(self):
        if os.path.exists(self.yunet_path) and os.path.exists(self.mobilefacenet_path):
            try:
                # Initialize YuNet
                self.detector = cv2.FaceDetectorYN.create(
                    self.yunet_path, "", (320, 320), 0.9, 0.3, 5000
                )
                # Initialize MobileFaceNet
                self.recognizer = cv2.dnn.readNetFromONNX(self.mobilefacenet_path)
                logger.info("Models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
        else:
            logger.error("ONNX models not found. Please run download_models.py")

    def _load_embeddings(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.names_file):
            try:
                self.known_embeddings = np.load(self.embeddings_file)
                with open(self.names_file, 'r') as f:
                    self.known_names = json.load(f)
                
                if len(self.known_embeddings) > 0:
                    self.is_trained = True
                    logger.info(f"Loaded {len(self.known_embeddings)} embeddings.")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
        else:
            logger.warning("No embeddings found. Dictionary will be empty.")

    def recognize_faces(self, frame):
        if self.detector is None:
            return [], []

        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))

        # Detect
        _, faces = self.detector.detect(frame)
        
        face_locations = []
        face_names = []

        if faces is not None:
            for face in faces:
                # YuNet returns [x, y, w, h, ...]
                box = face[:4].astype(int)
                x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                
                # Ensure bounds
                x = max(0, x)
                y = max(0, y)
                w_box = min(w_box, w - x)
                h_box = min(h_box, h - y)

                if w_box <= 0 or h_box <= 0:
                    continue

                # Prepare return format (top, right, bottom, left) used by main.py
                top, right, bottom, left = y, x + w_box, y + h_box, x
                face_locations.append((top, right, bottom, left))

                name = "Unknown"
                if self.is_trained:
                    try:
                        face_img = frame[y:y+h_box, x:x+w_box]
                        
                        # Preprocess
                        blob = cv2.dnn.blobFromImage(face_img, 1.0/128.0, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                        self.recognizer.setInput(blob)
                        embedding = self.recognizer.forward()
                        
                        # Normalize
                        embedding_norm = cv2.normalize(embedding, None, alpha=1, beta=0, norm_type=cv2.NORM_L2).flatten()
                        
                        # Compare with known
                        # Dot product of normalized vectors = Cosine Similarity
                        scores = np.dot(self.known_embeddings, embedding_norm)
                        best_idx = np.argmax(scores)
                        max_score = scores[best_idx]
                        
                        # Threshold (Tunable: 0.5 is safe, 0.6 is stricter)
                        if max_score > 0.5:
                            name = self.known_names[best_idx]
                            # name = f"{name} ({max_score:.2f})" # Debug
                        
                    except Exception as e:
                        logger.error(f"Inference error: {e}")

                face_names.append(name)

        return face_locations, face_names
