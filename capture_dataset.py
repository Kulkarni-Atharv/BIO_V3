
import cv2
import os

def capture_faces():
    print("--- Face Description ---")
    face_id = input("Enter User ID (numeric, e.g., 1): ")
    face_name = input("Enter User Name (e.g., Atharv): ")
    
    # Create directory
    dir_name = f"known_faces/{face_id}_{face_name}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Try GStreamer pipeline for Libcamera on Raspberry Pi
    gst_pipeline = (
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink"
    )
    cam = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        print("[WARN] GStreamer pipeline failed, falling back to index 0...")
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
    
    # Load YuNet Face Detector
    # path relative to where script is run, usually project root
    yunet_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(yunet_path):
        print(f"[ERROR] YuNet model not found at {yunet_path}. Please run download_models.py.")
        return

    # Initialize FaceDetectorYN
    # input_size will be set after reading the first frame
    detector = cv2.FaceDetectorYN.create(
        yunet_path,
        "",
        (320, 320), # Initial size, will be updated
        0.9, # Score threshold
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
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()
