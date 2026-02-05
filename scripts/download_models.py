
import os
import requests
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import ASSETS_DIR

def download_file(url, params=None):
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    filename = url.split('/')[-1]
    if "?raw=true" in filename:
        filename = filename.replace("?raw=true", "")
    
    local_path = os.path.join(ASSETS_DIR, filename)
    
    print(f"Downloading {filename} from {url}...")
    
    try:
        with requests.get(url, stream=True, params=params) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print(f"Downloaded: {local_path}")
        return local_path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

def main():
    # Model URLs
    # 1. Face Detection: YuNet
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    
    # 2. Face Recognition: MobileFaceNet
    # Using a known repository that hosts the ONNX converted model
    mobilefacenet_url = "https://github.com/foamliu/MobileFaceNet/blob/master/weights/MobileFaceNet.onnx?raw=true"
    
    print("--- Downloading Models ---")
    
    f1 = download_file(yunet_url)
    f2 = download_file(mobilefacenet_url)
    
    if f1 and f2:
        print("\nAll models downloaded successfully!")
    else:
        print("\nSome downloads failed. Please check internet connection or URLs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
