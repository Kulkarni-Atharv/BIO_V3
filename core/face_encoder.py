import cv2
import os
import sys
import numpy as np
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Encoder")

from shared.config import (
    YUNET_PATH, MOBILEFACENET_PATH, 
    EMBEDDINGS_FILE, NAMES_FILE, KNOWN_FACES_DIR,
    DETECTION_THRESHOLD
)

# [NEW] Import Aligner
try:
    from core.alignment import StandardFaceAligner
    aligner = StandardFaceAligner()
except ImportError:
    logger.warning("Could not import StandardFaceAligner. Using fallback cropping.")
    aligner = None

class FaceEncoder:
    def __init__(self):
        self.yunet_path = YUNET_PATH
        self.mobilefacenet_path = MOBILEFACENET_PATH
        self.embeddings_file = EMBEDDINGS_FILE
        self.names_file = NAMES_FILE
        self.known_faces_dir = KNOWN_FACES_DIR
        
        self.detector = None
        self.recognizer = None
        self.known_embeddings = []
        self.known_names = []
        
        self._load_models()
        self._load_existing_data()

    def _load_models(self):
        if not os.path.exists(self.yunet_path) or not os.path.exists(self.mobilefacenet_path):
            logger.error("Models not found. Run download_models.py first.")
            return

        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path, "", (320, 320), DETECTION_THRESHOLD, 0.3, 5000
        )
        self.recognizer = cv2.dnn.readNetFromONNX(self.mobilefacenet_path)

    # ... (rest of class remains same until _process_single_image) ...

    def _process_single_image(self, img_path):
        folder_name = os.path.basename(os.path.dirname(img_path))
        if "_" in folder_name:
            parts = folder_name.split('_')
            user_name = parts[1] # ID is parts[0]
        else:
            user_name = folder_name

        img = cv2.imread(img_path)
        if img is None: return None, None

        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        
        if faces is None or len(faces) == 0:
            return None, None

        # Take largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        
        # Landmarks are at index 4 to 13 (5 points x 2 coords)
        # x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
        landmarks = face[4:14].reshape((5, 2))
        
        face_img = None
        
        # Attempt Alignment
        if aligner:
            try:
                face_img = aligner.align(img, landmarks)
            except Exception as e:
                logger.warning(f"Alignment failed for {img_path}: {e}")
        
        # Fallback to cropping if alignment failed or not available
        if face_img is None:
            box = face[:4].astype(int)
            x, y, w_box, h_box = box[0], box[1], box[2], box[3]
            
            # Margin
            x = max(0, x); y = max(0, y)
            w_box = min(w_box, w - x); h_box = min(h_box, h - y)
            
            if w_box <= 0 or h_box <= 0: return None, None
            face_img = img[y:y+h_box, x:x+w_box]

        blob = cv2.dnn.blobFromImage(face_img, 1.0/128.0, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        self.recognizer.setInput(blob)
        embedding = self.recognizer.forward()
        embedding_norm = cv2.normalize(embedding, None, alpha=1, beta=0, norm_type=cv2.NORM_L2)
        
        return embedding_norm.flatten(), user_name

def main():
    encoder = FaceEncoder()
    encoder.process_images()

if __name__ == "__main__":
    main()
