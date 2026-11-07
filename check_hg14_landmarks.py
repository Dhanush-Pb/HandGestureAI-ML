from pathlib import Path

import cv2
import numpy as np

from landmark_utils import load_landmark_model, IMG_SIZE, CONNECTIONS

BASE = Path(__file__).parent
ROOT = BASE / "data" / "data_set" / "HG14-Hand Gesture"
OUT = BASE / "hg14_decode_check.png"

PAD = 0.4      # border, same as check_hg14_landmarks.py
TILE = 160

# gestures to look at (folder number: name)
SHOW = {1: "one", 2: "peace", 3: "three", 4: "four", 6: "call_me", 12: "rock"}


def decode(heat, beta=1.0, radius=None):
    """Heat maps (H, W, 21) -> 21 points in 0..1.

    beta   : higher = sticks to the strongest peak (1 = what you use now)
    radius : if set, only look at cells near the strongest peak
    """
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


VARIANTS = [
    ("now", dict(beta=1.0)),
    ("beta4", dict(beta=4.0)),
    ("peak", dict(beta=4.0, radius=3)),
]

model = load_landmark_model()
rows = []

for idx, name in SHOW.items():
    folder = ROOT / f"Gesture_{idx}"
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 10 ** 9,
    )
    picks = [files[int(len(files) * 0.2)], files[int(len(files) * 0.7)]]

    tiles = []
    for path in picks:
        img = cv2.imread(str(path))
        pad = int(img.shape[0] * PAD)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype("float32")
        heat = model(rgb[np.newaxis], training=False).numpy()[0]

        for label, kwargs in VARIANTS:
            pts = (decode(heat, **kwargs) * IMG_SIZE).astype(int)
            canvas = small.copy()
            for a, b in CONNECTIONS:
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 1)
            for p in pts:
                cv2.circle(canvas, tuple(p), 3, (0, 0, 255), -1)
            tile = cv2.resize(canvas, (TILE, TILE))
            cv2.putText(tile, f"{name} {label}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(tile)

    rows.append(np.hstack(tiles))

cv2.imwrite(str(OUT), np.vstack(rows))
print("Saved", OUT.name)