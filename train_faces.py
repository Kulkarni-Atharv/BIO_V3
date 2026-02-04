
import cv2
import os
import numpy as np

# Path for face image database
path = 'known_faces'

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Get the image paths and face samples
    image_paths = []
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("No images found in 'known_faces' folder.")
        print("Structure should be: known_faces/1_Name/image.jpg")
        return

    faceSamples = []
    ids = []
    names = {}

    print(f"Training on {len(image_paths)} images...")

    for imagePath in image_paths:
        try:
            PIL_img = cv2.imread(imagePath)
            img = cv2.cvtColor(PIL_img, cv2.COLOR_BGR2GRAY)
            
            # Extract ID from folder name. Expected format: known_faces/1/image.jpg or known_faces/1_Atharv/image.jpg
            folder_name = os.path.basename(os.path.dirname(imagePath))
            
            # Simple parsing: try to get the integer ID from the start of the folder name
            try:
                if "_" in folder_name:
                    id_str = folder_name.split('_')[0]
                    name_str = folder_name.split('_')[1]
                    id = int(id_str)
                    names[id] = name_str
                else:
                    id = int(folder_name)
                    names[id] = folder_name
            except ValueError:
                print(f"Skipping {imagePath}: Folder name '{folder_name}' must start with a number (ID). E.g., '1_John'")
                continue

            faces = detector.detectMultiScale(img)

            for (x,y,w,h) in faces:
                faceSamples.append(img[y:y+h,x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")

    if not ids:
        print("No valid faces found to train.")
        return

    recognizer.train(faceSamples, np.array(ids))

    # Save the model
    recognizer.write('trainer.yml')
    
    # Save the ID-to-Name mapping
    import json
    with open('names.json', 'w') as f:
        json.dump(names, f)

    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program")

if __name__ == "__main__":
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created '{path}' directory. Please add subfolders like '{path}/1_John' with images.")
    else:
        train_model()
