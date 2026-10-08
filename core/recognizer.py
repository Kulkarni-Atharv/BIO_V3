import cv2
import os
import json
import logging
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Recognizer")

from shared.config import (
    YUNET_PATH, MOBILEFACENET_PATH, 
    EMBEDDINGS_FILE, NAMES_FILE,
    DETECTION_THRESHOLD, RECOGNITION_THRESHOLD
)

# Width used for face detection during recognition (frame is downscaled to this)
DETECT_WIDTH = 320

# [NEW] Import Aligner
try:
    from core.alignment import StandardFaceAligner
    aligner = StandardFaceAligner()
except ImportError:
    aligner = None


class FaceRecognizer:
    def __init__(self):
        self.yunet_path = YUNET_PATH
        self.mobilefacenet_path = MOBILEFACENET_PATH
        self.embeddings_file = EMBEDDINGS_FILE
        self.names_file = NAMES_FILE
        
        self.detector = None
        self.recognizer = None
        self.known_embeddings = []
        self.known_names = []
        
        self._load_models()
        self._load_database()

    def _load_models(self):
        if not os.path.exists(self.yunet_path) or not os.path.exists(self.mobilefacenet_path):
            logger.error("Models not found.")
            return

        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path, "", (320, 320), DETECTION_THRESHOLD, 0.3, 5000
        )
        self.recognizer = cv2.dnn.readNetFromONNX(self.mobilefacenet_path)
        logger.info("Models loaded successfully.")

    def _load_database(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.names_file):
            try:
                self.known_embeddings = np.load(self.embeddings_file)
                with open(self.names_file, 'r') as f:
                    self.known_names = json.load(f)
                logger.info(f"Loaded {len(self.known_embeddings)} identities.")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.known_embeddings = []
                self.known_names = []
        else:
            logger.warning("No database found.")

    def recognize_faces(self, frame):
        if self.detector is None or self.recognizer is None:
            return [], []

        h, w, _ = frame.shape
        # Detect on a smaller copy (much faster on the Pi), then scale boxes/landmarks back
        scale = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))

        _, faces = self.detector.detect(small)
        if faces is not None and scale < 1.0:
            faces = faces.copy()
            faces[:, :14] /= scale
        
        face_locations = []
        face_names = []

        if faces is not None:
            for face in faces:
                # Bounding Box
                box = face[:4].astype(int)
                x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                
                # Landmarks for alignment
                landmarks = face[4:14].reshape((5, 2))
                
                face_locations.append((x, y, w_box, h_box))
                
                # Alignment
                # Alignment
                face_img = None
                name = "Unknown"
                
                if aligner:
                    try:
                        face_img = aligner.align(frame, landmarks)
                        if face_img is not None:
                            # MobileFaceNet expects RGB. aligner returns same as input (BGR if frame is BGR).
                            # We need to ensure blobFromImage parameters are correct.
                            # blobFromImage takes BGR if swapRB=True --> RGB.
                            # Our previous code used `swapRB=True`.
                            
                            # Get embedding
                            blob = cv2.dnn.blobFromImage(face_img, 1.0/128.0, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                            self.recognizer.setInput(blob)
                            embedding = self.recognizer.forward()
                            embedding_norm = cv2.normalize(embedding, None, alpha=1, beta=0, norm_type=cv2.NORM_L2)
                            
                            # Compare
                            if len(self.known_embeddings) > 0:
                                scores = np.dot(self.known_embeddings, embedding_norm.T).flatten()
                                best_match_idx = np.argmax(scores)
                                max_score = scores[best_match_idx]
                                
                                if max_score > RECOGNITION_THRESHOLD:
                                    name = self.known_names[best_match_idx]
                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                else:
                    # Fallback (same as original logic but inside else)
                    pass 

                face_names.append(name)

        return face_locations, face_names
