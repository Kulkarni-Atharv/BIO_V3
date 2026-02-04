
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
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")
    
    count = 0
    while(True):
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            
            # Save the captured image into the datasets folder
            # Save as grayscale for better training
            cv2.imwrite(f"{dir_name}/User.{face_id}.{count}.jpg", gray[y:y+h,x:x+w])
            
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
