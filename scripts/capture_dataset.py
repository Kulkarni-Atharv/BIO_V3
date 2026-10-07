
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import KNOWN_FACES_DIR, YUNET_PATH, DETECTION_THRESHOLD
from device.camera import open_usb_camera

def capture_faces():
    print("--- Face Description ---")
    face_id = input("Enter User ID (numeric, e.g., 1): ")
    face_name = input("Enter User Name (e.g., Atharv): ")
    
    # Create directory (Use shared path)
    dir_name = os.path.join(KNOWN_FACES_DIR, f"{face_id}_{face_name}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    cam = open_usb_camera()
    if not cam.isOpened():
        return

    try:
        # Load YuNet Face Detector
        if not os.path.exists(YUNET_PATH):
            print(f"[ERROR] YuNet model not found at {YUNET_PATH}. Please run scripts/download_models.py.")
            return
            
        detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320), # Initial size, will be updated
            DETECTION_THRESHOLD, # Score threshold
            0.3, # NMS threshold
            5000 # Top K
        )
        
        print("\n[INFO] Initializing face capture. Look at the camera and wait...")
        
        count = 0
        first_frame = True
        
        while(True):
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            height, width, _ = img.shape
            
            # Update detector input size if it's the first frame
            if first_frame:
                detector.setInputSize((width, height))
                first_frame = False

            # YuNet Detection
            # result is (faces, detections). detections is a list of [x, y, w, h, ...]
            _, faces = detector.detect(img)
            
            if faces is not None:
                for face in faces:
                    # YuNet returns float coordinates
                    x, y, w, h = map(int, face[:4])
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 0 and h > 0:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                        count += 1
                        
                        # Save the captured image into the datasets folder
                        cv2.imwrite(f"{dir_name}/User.{face_id}.{count}.jpg", img[y:y+h,x:x+w])
                        
                        cv2.imshow('image', img)
                
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                 break
            
        print(f"\n[INFO] Exiting Program and cleanup stuff. Captured {count} images in '{dir_name}'")
    
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
