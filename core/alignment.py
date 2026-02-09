
import cv2
import numpy as np

class StandardFaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=112, desiredFaceHeight=112):
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        
        # Standard landmarks for 112x112 arcface (approx)
        # 5 points: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
        self.reference_pts = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        
        if desiredFaceWidth == 112:
             self.reference_pts[:,0] += 8.0 # Shift slightly if needed, but standard 112x112 usually works directly
        
        # Actually, for 112x112 MobileFaceNet, the standard points are often:
        # src_pts = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        # Let's stick to a robust estimation.

    def align(self, image, landmarks):
        """
        Aligns the face image using the provided 5 landmarks.
        landmarks: numpy array of shape (5, 2)
        """
        if landmarks is None or len(landmarks) != 5:
            return None
        
        # Convert landmarks to float32
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Estimate the affine transformation matrix (Similarity Transform)
        # We map the DETECTED landmarks to the REFERENCE landmarks
        # landmarks -> reference_pts
        
        # Using skimage.transform.SimilarityTransform equivalent in OpenCV
        # cv2.estimateAffinePartial2D is robust for this
        tform, _ = cv2.estimateAffinePartial2D(landmarks, self.reference_pts)
        
        if tform is None:
            return None

        # Apply the transformation (warp)
        output_size = (self.desiredFaceWidth, self.desiredFaceHeight)
        aligned_face = cv2.warpAffine(image, tform, output_size)
        
        return aligned_face

# Default instance
aligner = StandardFaceAligner()
