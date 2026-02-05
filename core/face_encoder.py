import cv2
import os
import numpy as np
import json
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Encoder")

from shared.config import (
    YUNET_PATH, MOBILEFACENET_PATH, 
    EMBEDDINGS_FILE, NAMES_FILE, KNOWN_FACES_DIR
)

class FaceEncoder:
    def __init__(self):
        self.yunet_path = YUNET_PATH
        self.mobilefacenet_path = MOBILEFACENET_PATH
        self.embeddings_file = EMBEDDINGS_FILE
        self.names_file = NAMES_FILE
        self.known_faces_dir = KNOWN_FACES_DIR
        
        self._load_models()
        self.known_embeddings = []
        self.known_names = []
        self.names_map = {}

    def _load_models(self):
        if not os.path.exists(self.yunet_path) or not os.path.exists(self.mobilefacenet_path):
            logger.error("Models not found. Run download_models.py first.")
            return

        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path, "", (320, 320), 0.9, 0.3, 5000
        )
        self.recognizer = cv2.dnn.readNetFromONNX(self.mobilefacenet_path)

    def process_images(self):
        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"No {self.known_faces_dir} directory found.")
            return False

        image_paths = []
        for root, dirs, files in os.walk(self.known_faces_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))


        if not image_paths:
            logger.warning("No images found. Clearing embeddings.")
            # Proceed to save empty files instead of returning False
            
        logger.info(f"Found {len(image_paths)} images. Processing...")
        self.known_embeddings = []
        self.known_names = []

        for img_path in image_paths:
            try:
                self._process_single_image(img_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Save (Always save, even if empty)
        np.save(self.embeddings_file, np.array(self.known_embeddings))
        with open(self.names_file, 'w') as f:
            json.dump(self.known_names, f)
        logger.info(f"Saved {len(self.known_embeddings)} embeddings.")
        return True

    def _process_single_image(self, img_path):
        folder_name = os.path.basename(os.path.dirname(img_path))
        if "_" in folder_name:
            parts = folder_name.split('_')
            user_id = parts[0]
            user_name = parts[1]
        else:
            user_id = folder_name
            user_name = folder_name

        img = cv2.imread(img_path)
        if img is None: return

        h, w, _ = img.shape
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        face_img = None
        
        if faces is None or len(faces) == 0:
            logger.warning(f"No face detected in {img_path}. Using full image as fallback.")
            face_img = img
        else:
            face = faces[0]
            box = face[:4].astype(int)
            x, y, w_box, h_box = box[0], box[1], box[2], box[3]
            
            x = max(0, x)
            y = max(0, y)
            w_box = min(w_box, w - x)
            h_box = min(h_box, h - y)
            
            if w_box > 0 and h_box > 0:
                face_img = img[y:y+h_box, x:x+w_box]
            else:
                face_img = img

        if face_img is None or face_img.size == 0: return

        blob = cv2.dnn.blobFromImage(face_img, 1.0/128.0, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        self.recognizer.setInput(blob)
        embedding = self.recognizer.forward()
        embedding_norm = cv2.normalize(embedding, None, alpha=1, beta=0, norm_type=cv2.NORM_L2)
        
        self.known_embeddings.append(embedding_norm.flatten())
        self.known_names.append(user_name)


def main():
    encoder = FaceEncoder()
    encoder.process_images()

if __name__ == "__main__":
    main()
