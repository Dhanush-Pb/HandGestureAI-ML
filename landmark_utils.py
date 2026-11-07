from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

BASE = Path(__file__).parent
LANDMARK_MODEL_PATH = BASE / "landmark_heatmap_v3.keras"
IMG_SIZE = 192            # must match training

# The hand picture is shrunk to this fraction and centred on a plain background.
ZOOM = 0.65
INNER = int(IMG_SIZE * ZOOM)
PAD = (IMG_SIZE - INNER) // 2
OFFSET = PAD / IMG_SIZE
SCALE = INNER / IMG_SIZE

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_landmark_model():
    return keras.models.load_model(LANDMARK_MODEL_PATH)


def decode(heat):
    """heat: (H, W, 21) logits -> 21 (x, y) points in 0..1."""
    h, w, k = heat.shape
    flat = heat.reshape(h * w, k)
    flat = flat - flat.max(axis=0)
    prob = np.exp(flat)
    prob /= prob.sum(axis=0)
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h
    gx = np.tile(xs, (h, 1)).reshape(h * w, 1)
    gy = np.tile(ys[:, None], (1, w)).reshape(h * w, 1)
    return np.stack([(prob * gx).sum(0), (prob * gy).sum(0)], axis=1)


def to_model_input(rgb):
    """Shrink an RGB picture and centre it on a plain background (192x192x3, float32)."""
    ring = np.concatenate([
        rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
        rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3),
    ])
    colour = np.median(ring, axis=0)
    small = cv2.resize(rgb, (INNER, INNER), interpolation=cv2.INTER_AREA)
    canvas = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    canvas[:] = colour
    canvas[PAD:PAD + INNER, PAD:PAD + INNER] = small
    return canvas


def guide_box(frame):
    """Square in the middle of the frame: returns x1, y1, side."""
    height, width = frame.shape[:2]
    side = int(min(height, width) * 0.8)
    return (width - side) // 2, (height - side) // 2, side


def find_points(model, frame):
    """Returns 21 points in 0..1, relative to the guide box."""
    x1, y1, side = guide_box(frame)
    crop = frame[y1:y1 + side, x1:x1 + side]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    heat = model(to_model_input(rgb)[np.newaxis], training=False).numpy()[0]
    return (decode(heat) - OFFSET) / SCALE


def draw_hand(frame, pts):
    x1, y1, side = guide_box(frame)
    px = (pts * side + np.array([x1, y1])).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), (0, 255, 0), 2)
    for p in px:
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (255, 255, 0), 2)


def normalize(pts):
    """Wrist becomes (0, 0); hand size becomes 1. Returns 42 numbers."""
    rel = pts - pts[0]
    scale = np.linalg.norm(rel, axis=1).max() + 1e-6
    return (rel / scale).astype("float32").reshape(-1)