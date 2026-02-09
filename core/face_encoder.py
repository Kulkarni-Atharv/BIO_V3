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

    def _load_existing_data(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.names_file):
            try:
                self.known_embeddings = [emb for emb in np.load(self.embeddings_file)]
                with open(self.names_file, 'r') as f:
                    self.known_names = json.load(f)
                logger.info(f"Loaded existing {len(self.known_embeddings)} embeddings.")
            except Exception as e:
                logger.error(f"Failed to load existing data: {e}. Starting fresh.")
                self.known_embeddings = []
                self.known_names = []

    def process_images(self):
        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"No {self.known_faces_dir} directory found.")
            return False

        current_images = {}
        new_images = []

        # 1. Scan directory
        for root, dirs, files in os.walk(self.known_faces_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(root, file)
                    # Use relpath as ID to be safe across machines/mounts
                    # But for now, absolute path changes if we move folder.
                    # Let's simple check if we have it? 
                    # Actually, simplistic incremental:
                    # just verify if this (name, image_hash) exists? Too slow.
                    # Better: Just check if we have enough embeddings for this user?
                    # Easiest Strategy for now:
                    # Clear lists and RE-DO everything? NO, that's what caused the bug.
                    
                    # Better Strategy: 
                    # We can't easily map embedding -> filename without a separate log.
                    # Since we don't have a log file yet, we will implement a robust full-scan
                    # that appends to a MEMORY list, then overwrites file.
                    # The "forgetting" happened because previous `known_embeddings = []` wiped it
                    # and `os.walk` likely failed to find old folders.
                    
                    # To fix "forgetting" IMMEDIATELY without complex log:
                    # We will re-scan everything. 
                    # If it's slow, we accept it for now (30 images is fast).
                    # But we must ensure `os.walk` finds everything.
                    
                    new_images.append(path)

        if not new_images:
            # Even if no new images, we must run Garbage Collection to remove deleted users
            pass

        # 2. Re-process ALL (Robust Fix) or Incremental?
        # User wants "1000 users". Full re-process is NOT viable.
        # We MUST do incremental.
        
        # We need a way to know if an image is already processed.
        # Since we don't have a DB for images, we can check if `names.json` contains the user?
        # No, a user has 30 images.
        
        # HYBRID APPROACH:
        # Load existing.
        # Process new images that are NOT in a "processed_log.json".
        
        processed_log_path = os.path.join(os.path.dirname(self.embeddings_file), "processed_images.json")
        processed_files = set()
        if os.path.exists(processed_log_path):
            with open(processed_log_path, 'r') as f:
                processed_files = set(json.load(f))

        # --- GARBAGE COLLECTION START ---
        # 1. Identify valid users (folders that exist)
        valid_users = set()
        for d in os.listdir(self.known_faces_dir):
            if os.path.isdir(os.path.join(self.known_faces_dir, d)):
                # Extract user name from folder "ID_Name" or "Name"
                if "_" in d:
                    valid_users.add(d.split('_')[1])
                else:
                    valid_users.add(d)
        
        # 2. Filter existing embeddings
        # We only have `known_names`. We must keep indices where name is in valid_users.
        # This is strictly for removing DELETED users.
        
        initial_count = len(self.known_names)
        filtered_embeddings = []
        filtered_names = []
        
        for emb, name in zip(self.known_embeddings, self.known_names):
             if name in valid_users:
                 filtered_embeddings.append(emb)
                 filtered_names.append(name)
        
        deleted_count = initial_count - len(filtered_names)
        if deleted_count > 0:
            logger.info(f"Garbage Collection: Removed {deleted_count} embeddings for deleted users.")
            # Update lists
            self.known_embeddings = filtered_embeddings
            self.known_names = filtered_names
        
        # --- GARBAGE COLLECTION END ---

        images_to_process = [img for img in new_images if img not in processed_files]
        
        if not images_to_process and deleted_count == 0:
            logger.info("No new images to process and no deleted users.")
            return True

        if images_to_process:
            logger.info(f"Found {len(images_to_process)} new images.")
        
        count = 0
        for img_path in images_to_process:
            try:
                emb, name = self._process_single_image(img_path)
                if emb is not None:
                    self.known_embeddings.append(emb)
                    self.known_names.append(name)
                    processed_files.add(img_path)
                    count += 1
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # 3. Save Updates
        if count > 0 or deleted_count > 0:
            np.save(self.embeddings_file, np.array(self.known_embeddings))
            with open(self.names_file, 'w') as f:
                json.dump(self.known_names, f)
            
            with open(processed_log_path, 'w') as f:
                json.dump(list(processed_files), f)
                
            logger.info(f"Changes saved. Added: {count}, Deleted: {deleted_count}. Total: {len(self.known_embeddings)}")
        
        return True

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
