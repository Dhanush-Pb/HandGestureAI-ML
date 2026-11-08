
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"

FREIHAND_DIR = DATA_DIR / "data_set" / "freihand"
HG14_DIR = DATA_DIR / "data_set" / "HG14-Hand Gesture"
GESTURE_CSV = DATA_DIR / "gestures_hg14.csv"

LANDMARK_MODEL = MODEL_DIR / "landmark_heatmap_v3.keras"
GESTURE_MODEL = MODEL_DIR / "gesture_model.keras"
GESTURE_LABELS = MODEL_DIR / "gesture_labels.json"
