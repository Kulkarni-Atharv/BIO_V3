"""
core/face_guide.py
------------------
Guided face registration.

Checks every camera frame for face position, size, lighting, sharpness and head
pose, tells the user what to change, and decides when a frame is good enough
to be saved as a training sample. Samples are collected in pose stages
(front, left, right) so the stored face data covers different angles.

Pure logic (no Qt / camera) - used by hmi.py VideoThread.
"""

import time

import cv2
import numpy as np

# Pose stages: (key, step title, instruction, samples to capture)
STAGES = [
    ("FRONT", "Front",      "Look directly at the camera",              12),
    ("LEFT",  "Left side",  "Turn your head slightly to your LEFT",      9),
    ("RIGHT", "Right side", "Turn your head slightly to your RIGHT",     9),
]

# Quality thresholds - tune on the device if needed
MIN_SCORE        = 0.80   # YuNet detection confidence
MIN_FACE_RATIO   = 0.22   # face width / frame width  -> below: "move closer"
MAX_FACE_RATIO   = 0.55   # face width / frame width  -> above: "move farther"
CENTER_TOLERANCE = 0.20   # max distance of face centre from frame centre (fraction of frame size)
MIN_BRIGHTNESS   = 60     # mean grey level of the face region
MAX_BRIGHTNESS   = 210
MIN_SHARPNESS    = 40.0   # variance of Laplacian on the 112x112 face
FRONT_MAX_YAW    = 0.15   # |yaw| allowed for the front stage
SIDE_MIN_YAW     = 0.22   # |yaw| needed for a left/right sample
SIDE_MAX_YAW     = 0.65   # |yaw| above this is turned too far
SAMPLE_INTERVAL  = 0.25   # seconds between saved samples (gives varied frames)
CROP_MARGIN      = 0.5    # extra context around the face box in saved images

# Minimum usable embeddings after training for registration to count as complete
MIN_VALID_SAMPLES = 20


def estimate_yaw(landmarks):
    """
    Head yaw from YuNet's 5 landmarks (unmirrored camera image).
    ~0 = frontal, > 0 = turned to the person's LEFT, < 0 = to the person's RIGHT.
    """
    eye_a, eye_b, nose = landmarks[0], landmarks[1], landmarks[2]
    left_x, right_x = min(eye_a[0], eye_b[0]), max(eye_a[0], eye_b[0])
    d_left = nose[0] - left_x      # nose to eye on image-left
    d_right = right_x - nose[0]    # nose to eye on image-right
    total = d_left + d_right
    if total <= 1:
        return 0.0
    return float((d_left - d_right) / total)


def crop_face(frame, face, margin=CROP_MARGIN):
    """Crop the face box with a margin so the encoder can re-detect it later."""
    h, w = frame.shape[:2]
    x, y, bw, bh = face[:4].astype(int)
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
    return frame[y1:y2, x1:x2]


class RegistrationGuide:
    def __init__(self):
        self.stage_idx = 0
        self.stage_count = 0
        self.total_count = 0
        self.total_target = sum(s[3] for s in STAGES)
        self.last_sample_time = 0.0

    # --- state ---
    @property
    def done(self):
        return self.stage_idx >= len(STAGES)

    @property
    def stage_key(self):
        return STAGES[min(self.stage_idx, len(STAGES) - 1)][0]

    @property
    def stage_text(self):
        if self.done:
            return "Done"
        _, title, _, target = STAGES[self.stage_idx]
        return f"Step {self.stage_idx + 1} of {len(STAGES)}: {title}  ({self.stage_count}/{target})"

    @property
    def instruction(self):
        return STAGES[min(self.stage_idx, len(STAGES) - 1)][2]

    @property
    def progress(self):
        return int(100 * self.total_count / self.total_target)

    # --- per-frame check ---
    def evaluate(self, frame, faces):
        """
        Returns (ok, message, face).
        ok      - True if this frame is a good sample for the current stage
        message - instruction to show the user
        face    - the YuNet face row used (None if no single face)
        """
        if faces is None or len(faces) == 0:
            return False, "No face detected - look at the camera", None
        if len(faces) > 1:
            return False, "Only one person in front of the camera", None

        face = faces[0]
        fh, fw = frame.shape[:2]
        x, y, bw, bh = face[:4]

        if face[14] < MIN_SCORE:
            return False, "Face not clear - look at the camera", face

        ratio = bw / fw
        if ratio < MIN_FACE_RATIO:
            return False, "Move your face closer", face
        if ratio > MAX_FACE_RATIO:
            return False, "Move your face farther away", face

        cx, cy = (x + bw / 2) / fw, (y + bh / 2) / fh
        if abs(cx - 0.5) > CENTER_TOLERANCE or abs(cy - 0.5) > CENTER_TOLERANCE:
            return False, "Keep your face in the centre of the circle", face

        roi = crop_face(frame, face, margin=0.0)
        if roi.size == 0:
            return False, "Keep your face inside the frame", face
        gray = cv2.cvtColor(cv2.resize(roi, (112, 112)), cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < MIN_BRIGHTNESS:
            return False, "Too dark - face the light", face
        if brightness > MAX_BRIGHTNESS:
            return False, "Too bright - avoid direct light", face
        if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
            return False, "Hold still", face

        yaw = estimate_yaw(face[4:14].reshape(5, 2))
        key = self.stage_key
        if key == "FRONT":
            if abs(yaw) > FRONT_MAX_YAW:
                return False, "Look directly at the camera", face
        else:
            wanted = yaw if key == "LEFT" else -yaw
            if wanted < SIDE_MIN_YAW:
                side = "LEFT" if key == "LEFT" else "RIGHT"
                return False, f"Turn your head slightly to your {side}", face
            if abs(yaw) > SIDE_MAX_YAW:
                return False, "Too far - turn back a little", face

        return True, "Hold still - capturing...", face

    def ready_for_sample(self, now=None):
        now = time.time() if now is None else now
        return now - self.last_sample_time >= SAMPLE_INTERVAL

    def accept_sample(self, now=None):
        """Count a saved sample; advance to the next stage when this one is full."""
        self.last_sample_time = time.time() if now is None else now
        self.stage_count += 1
        self.total_count += 1
        if self.stage_count >= STAGES[self.stage_idx][3]:
            self.stage_idx += 1
            self.stage_count = 0
