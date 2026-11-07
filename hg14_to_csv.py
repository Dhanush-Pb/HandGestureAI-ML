import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from landmark_utils import load_landmark_model, decode, normalize, to_model_input

BASE = Path(__file__).parent
ROOT = BASE / "data" / "data_set" / "HG14-Hand Gesture"
OUT = BASE / "data" / "gestures_hg14.csv"
BLOCK = 100      # 100 consecutive images count as one "take"
BATCH = 32

NAMES = {
    0: "fist", 1: "one", 2: "peace", 3: "three", 4: "four", 5: "five",
    6: "call_me", 7: "pinch", 8: "l_shape", 9: "thumb_side",
    10: "point_b", 11: "love_you", 12: "rock", 13: "flat_hand",
}

# Only these 7 gestures are used.
KEEP = {"fist", "five", "flat_hand", "call_me", "love_you", "thumb_side", "pinch"}

model = load_landmark_model()
run = tf.function(lambda x: model(x, training=False))


def read_batch(paths):
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(to_model_input(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return imgs


def sort_key(p):
    return int(p.stem) if p.stem.isdigit() else 10**9


with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "take"] + [f"p{i}" for i in range(42)])

    for idx in range(14):
        name = NAMES.get(idx)
        if name is None or (KEEP and name not in KEEP):
            continue
        folder = ROOT / f"Gesture_{idx}"
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=sort_key,
        )
        rows = 0
        for start in range(0, len(files), BATCH):
            chunk = files[start:start + BATCH]
            imgs = read_batch(chunk)
            if not imgs:
                continue
            heat = run(tf.constant(np.array(imgs, dtype="float32"))).numpy()
            for k, h in enumerate(heat):
                take = 1000 + idx * 100 + (start + k) // BLOCK
                writer.writerow([name, take] + normalize(decode(h)).tolist())
                rows += 1
        print(f"{name}: {rows} rows")

print("Saved", OUT)