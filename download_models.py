import os
import requests
import sys

def download_file(url, params=None):
    local_filename = url.split('/')[-1]
    # Handle GitHub raw/blob links
    if "?raw=true" in local_filename:
        local_filename = local_filename.replace("?raw=true", "")
    
    # If the user provided a filename in the query param (unlikely here but good practice), just use basename
    print(f"Downloading {local_filename} from {url}...")
    
    try:
        with requests.get(url, stream=True, params=params) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print(f"Downloaded: {local_filename}")
        return local_filename
    except Exception as e:
        print(f"Failed to download {local_filename}: {e}")
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
