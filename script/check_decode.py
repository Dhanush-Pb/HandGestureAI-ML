import json
from pathlib import Path

import cv2
import numpy as np

from landmark_utils import load_landmark_model, IMG_SIZE

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "data_set"
IMAGE_DIR = DATA_DIR / "freihand" / "rgb"

SRC = 224          # FreiHAND image size
N_CHECK = 600      # how many validation images to test
BATCH = 32
FINGERTIPS = [4, 8, 12, 16, 20]


def decode(heat, beta=1.0, radius=None):
    h, w, k = heat.shape
    flat = heat.reshape(h * w, k)
    prob = np.exp(beta * (flat - flat.max(axis=0)))

    if radius is not None:
        row, col = np.divmod(np.arange(h * w), w)
        peak_row, peak_col = np.divmod(flat.argmax(axis=0), w)
        near = (np.abs(row[:, None] - peak_row[None, :]) <= radius) & \
               (np.abs(col[:, None] - peak_col[None, :]) <= radius)
        prob = prob * near

    prob = prob / prob.sum(axis=0)
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h
    gx = np.tile(xs, h)[:, None]
    gy = np.repeat(ys, w)[:, None]
    return np.stack([(prob * gx).sum(0), (prob * gy).sum(0)], axis=1)


# --------------------------------------------------
# True points (same as in training)
# --------------------------------------------------
with open(DATA_DIR / "training_xyz.json") as f:
    xyz_all = np.array(json.load(f), dtype=np.float32)
with open(DATA_DIR / "training_K.json") as f:
    k_all = np.array(json.load(f), dtype=np.float32)
n_hands = len(xyz_all)


def project(xyz, K):
    uv = (K @ xyz.T).T
    return uv[:, :2] / uv[:, 2:3]


image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
hand_of = np.arange(len(image_paths)) % n_hands
val_pool = np.where(hand_of % 10 == 0)[0]      # hands the model never trained on
chosen = np.random.default_rng(0).choice(val_pool, size=N_CHECK, replace=False)

truth = np.stack([project(xyz_all[hand_of[i]], k_all[hand_of[i]]) for i in chosen])

# --------------------------------------------------
# Run the model
# --------------------------------------------------
model = load_landmark_model()
heats = []
for start in range(0, N_CHECK, BATCH):
    batch = []
    for i in chosen[start:start + BATCH]:
        img = cv2.imread(str(image_paths[i]))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    heats.append(model(np.array(batch, dtype="float32"), training=False).numpy())
heats = np.concatenate(heats)

# --------------------------------------------------
# Compare the three ways of reading the heat maps
# --------------------------------------------------
variants = [
    ("now   (beta 1, whole map)", dict(beta=1.0)),
    ("beta4 (beta 4, whole map)", dict(beta=4.0)),
    ("peak  (beta 4, near peak)", dict(beta=4.0, radius=3)),
]

print(f"\nError in pixels on a {SRC}x{SRC} image, {N_CHECK} FreiHAND validation images")
print("(distance between the true point and the predicted point)\n")
print(f"{'method':28s} {'all points':>11s} {'fingertips':>11s}")

for name, kwargs in variants:
    pred = np.stack([decode(h, **kwargs) for h in heats]) * SRC
    dist = np.linalg.norm(pred - truth, axis=2)          # (N, 21)
    print(f"{name:28s} {dist.mean():11.2f} {dist[:, FINGERTIPS].mean():11.2f}")