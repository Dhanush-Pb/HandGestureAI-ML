from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from common.paths import LANDMARK_MODEL

MODEL_PATH = LANDMARK_MODEL
IMG_SIZE = 192            # must match training
SMOOTHING = 0.5           # 0 = no smoothing, closer to 1 = steadier but slower

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

model = keras.models.load_model(MODEL_PATH)


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


camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")
print("Show your hand inside the square. Press Q to quit.")

previous = None
while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    side = int(min(height, width) * 0.8)
    x1, y1 = (width - side) // 2, (height - side) // 2
    crop = frame[y1:y1 + side, x1:x1 + side]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype("float32")

    heat = model(small[np.newaxis], training=False).numpy()[0]
    pts = decode(heat)

    smoothed = pts if previous is None else SMOOTHING * previous + (1 - SMOOTHING) * pts
    previous = smoothed

    px = (smoothed * side + np.array([x1, y1])).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), (0, 255, 0), 2)
    for p in px:
        cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (255, 255, 0), 2)
    cv2.imshow("Landmark model v3", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()