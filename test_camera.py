import cv2

def test_gst():
    print("\n--- Testing Ribbon Cable (CSI) Camera via GStreamer ---")
    gst_pipeline = (
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink"
    )
    print(f"Pipeline: {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[FAIL] GStreamer pipeline failed to open. Missing plugins?")
        return
    
    ret, frame = cap.read()
    if ret:
        print(f"[SUCCESS] Ribbon Camera Working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("[FAIL] Pipeline opened but no frame received.")
    cap.release()

def list_ports():
    print("\n--- Testing USB Cameras (V4L2) ---")
    is_working = True
    dev_port = 0
    working_ports = []
    
    # Only test ports 0-5 to avoid indefinite hangs
    while dev_port < 5:
        camera = cv2.VideoCapture(dev_port, cv2.CAP_V4L2)
        if not camera.isOpened():
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
            camera.release()
        dev_port +=1

if __name__ == "__main__":
    test_gst()
    list_ports()
