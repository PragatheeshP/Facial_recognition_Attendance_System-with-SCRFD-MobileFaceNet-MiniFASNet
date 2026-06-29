"""
Central configuration for the lightweight attendance system.

Everything here is tuned for a small classroom (roughly 2-10 students,
i.e. Tier 2 from the architecture doc) running entirely on CPU.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Storage ---
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
DB_URL = f"sqlite:///{DB_PATH}"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "face_embeddings.index")

# --- Face engine (InsightFace) ---
# buffalo_s = SCRFD-500MF detector + MobileFaceNet recognizer, ~13MB combined.
# Swap to "buffalo_l" only if you later need max accuracy and have a GPU.
INSIGHTFACE_MODEL_PACK = "buffalo_s"
DETECTION_SIZE = (320, 320)  # smaller than the 640x640 default; fine for 5 students close to camera
EMBEDDING_DIM = 512

# --- Recognition ---
# Cosine similarity threshold for "same person". Start at 0.40 (matches the
# original doc's ArcFace guidance) and tune up/down after a few real sessions.
RECOGNITION_THRESHOLD = 0.40
NUM_ENROLLMENT_IMAGES = 5  # frontal, slight-left, slight-right, glasses-if-any, different lighting

# --- Proximity / frame-fill check ---
# If a detected face box occupies more than this fraction of the total frame
# area, we reject it as "too close" — a common attack vector is holding a
# printed photo right up against the camera lens.  Also catches genuine
# students who are simply too close, in which case the UI tells them to step back.
#
# Frame area = width * height of the captured image.
# Face area  = (x2-x1) * (y2-y1) of the detected bounding box.
# Reject if: face_area / frame_area > MAX_FACE_FILL_RATIO
#
# Tuning guidance:
#   0.35  (~60% of frame width/height)  — fairly strict, good for anti-spoof
#   0.50  (~70% of frame width/height)  — lenient, may let close-up photos through
#   0.25  (~50% of frame width/height)  — very strict, may reject normal seating
MAX_FACE_FILL_RATIO = 0.35

# --- Liveness ---
# Real passive anti-spoofing via MiniFASNet (Silent-Face-Anti-Spoofing),
# converted from the official PyTorch checkpoints to ONNX so inference only
# needs onnxruntime (already a dependency) — no torch at runtime.
# See src/liveness.py for how the two-model ensemble works.
ENABLE_LIVENESS = True

ANTI_SPOOF_MODEL_DIR = os.path.join(BASE_DIR, "resources", "anti_spoof_models")
ANTI_SPOOF_MODELS = [
    # (filename, crop_scale) — crop_scale controls how much context around
    # the tight face box each model sees; matches the official repo's
    # naming convention (scale is encoded in the filename).
    ("2.7_80x80_MiniFASNetV2.onnx", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0),
]

# Probability mass assigned to the "real" class, averaged across both
# models in the ensemble (range 0.0-1.0). 0.70 is the original repo's own
# default operating point; raise it if you see real students getting
# rejected, lower it if photos/screens are still slipping through.
LIVENESS_THRESHOLD = 0.65

# --- Periodic attendance re-check ---
# To prevent "mark and leave" cheating, the system can automatically
# re-verify presence every N minutes during an active session.
# When enabled, a background timer triggers a new snapshot + recognition
# pass. Students who were previously marked PRESENT but are not detected
# in the periodic check get flagged as LEFT.
#
# PERIODIC_CHECK_INTERVAL_MINUTES = 0  disables automatic periodic checks.
# Set to 5 (or another value) to enable.
PERIODIC_CHECK_INTERVAL_MINUTES = 1

# How long after the last successful detection a student is considered
# "still present" before being marked LEFT.  This grace period accounts
# for momentary occlusion (someone walking in front of the camera,
# student looking away, etc.).
# Should be >= PERIODIC_CHECK_INTERVAL_MINUTES so a single missed check
# doesn't immediately flag everyone as gone.
PRESENCE_GRACE_PERIOD_MINUTES = 7

# --- Attendance logic ---
DEDUP_WINDOW_SECONDS = 60  # don't re-mark the same student twice within this window
RE_ENROLL_ACCURACY_THRESHOLD = 0.85  # flag for re-enrollment if avg confidence drops below this