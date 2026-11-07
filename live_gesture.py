import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from landmark_utils import (load_landmark_model, to_model_input, decode,
                            guide_box, normalize, OFFSET, SCALE, CONNECTIONS)

BASE = Path(__file__).parent

# ---------------- settings ----------------
CONFIDENCE = 0.70      # at or above this = "TARGET LOCKED"
SKIP = 2               # run the models every 2nd frame (1 = every frame, slower)
SMOOTH = 6             # average the last few predictions
VIEW_H = 600           # height of the video area (lower, e.g. 480, if FPS is low)
PANEL_W = 340          # width of the right-hand panel
TITLE = "HAND-NET // GESTURE SCANNER"

# ---------------- colours (BGR) ----------------
CYAN = (255, 230, 0)
GREEN = (120, 255, 60)
AMBER = (0, 180, 255)
RED = (80, 80, 255)
MAGENTA = (255, 60, 255)
WHITE = (240, 240, 240)
DIM = (150, 125, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, text, org, scale=0.5, color=CYAN, thick=1):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def brackets(img, x1, y1, x2, y2, color, length, thick=2):
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * length, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * length), color, thick, cv2.LINE_AA)


def dim_colour(c, k=0.35):
    return tuple(int(v * k) for v in c)


def make_panel_background(height):
    bg = np.full((height, PANEL_W, 3), (24, 17, 10), dtype=np.uint8)
    for x in range(0, PANEL_W, 30):
        cv2.line(bg, (x, 0), (x, height), (38, 28, 18), 1)
    for y in range(0, height, 30):
        cv2.line(bg, (0, y), (PANEL_W, y), (38, 28, 18), 1)
    cv2.line(bg, (0, 0), (0, height), CYAN, 2)
    return bg


# ---------------- models ----------------
landmarks = load_landmark_model()
fast = tf.function(lambda x: landmarks(x, training=False))
classifier = keras.models.load_model(BASE / "gesture_model.keras")
with open(BASE / "gesture_labels.json") as f:
    labels = json.load(f)
print("Gestures:", labels)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")
print("Q = quit, S = save a screenshot")

recent = deque(maxlen=SMOOTH)
pts = None
avg = np.zeros(len(labels))
latency = 0.0
fps = 0.0
frame_no = 0
typed = 0.0
last_text = ""
panel_bg = None
start = time.time()
last = start
WIN = "HAND-NET // GESTURE SCANNER"

while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h0, w0 = frame.shape[:2]
    x1, y1, side = guide_box(frame)

    # ---- models (every SKIP-th frame) ----
    if frame_no % SKIP == 0 or pts is None:
        t0 = time.time()
        crop = frame[y1:y1 + side, x1:x1 + side]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        heat = fast(tf.constant(to_model_input(rgb)[np.newaxis])).numpy()[0]
        pts = (decode(heat) - OFFSET) / SCALE
        probs = classifier(normalize(pts)[np.newaxis], training=False).numpy()[0]
        recent.append(probs)
        avg = np.mean(recent, axis=0)
        latency = (time.time() - t0) * 1000
    frame_no += 1

    top = int(avg.argmax())
    conf = float(avg[top])
    if conf >= CONFIDENCE:
        status, status_col, name = "TARGET LOCKED", GREEN, labels[top]
    elif conf >= 0.45:
        status, status_col, name = "ANALYZING...", AMBER, "???"
    else:
        status, status_col, name = "SCANNING...", RED, "NO LOCK"
    locked = status == "TARGET LOCKED"
    shown = name.upper().replace("_", " ")

    # ---- video area ----
    S = VIEW_H / h0
    view = cv2.resize(frame, (int(w0 * S), VIEW_H))
    view = cv2.convertScaleAbs(view, alpha=0.82, beta=-8)      # darker, techno look
    tint = np.full_like(view, CYAN)
    view = cv2.addWeighted(tint, 0.06, view, 0.94, 0)
    VW = view.shape[1]

    X1, Y1, SIDE = int(x1 * S), int(y1 * S), int(side * S)
    brackets(view, X1, Y1, X1 + SIDE, Y1 + SIDE, CYAN, 28, 2)

    # scanning line inside the guide box
    t = time.time()
    sy = Y1 + int(((t * 0.55) % 1.0) * SIDE)
    ya, yb = max(Y1, sy - 8), min(Y1 + SIDE, sy + 8)
    if yb > ya:
        strip = view[ya:yb, X1:X1 + SIDE]
        glow = cv2.addWeighted(np.full_like(strip, CYAN), 0.28, strip, 0.72, 0)
        view[ya:yb, X1:X1 + SIDE] = glow
        cv2.line(view, (X1, sy), (X1 + SIDE, sy), CYAN, 1, cv2.LINE_AA)

    # skeleton
    px = ((pts * side + np.array([x1, y1])) * S).astype(int)
    for a, b in CONNECTIONS:
        cv2.line(view, tuple(px[a]), tuple(px[b]), dim_colour(CYAN), 7, cv2.LINE_AA)
    for a, b in CONNECTIONS:
        cv2.line(view, tuple(px[a]), tuple(px[b]), CYAN, 2, cv2.LINE_AA)
    for i, p in enumerate(px):
        cv2.circle(view, tuple(p), 4, WHITE, -1, cv2.LINE_AA)
        if i in (4, 8, 12, 16, 20):
            cv2.circle(view, tuple(p), 9, MAGENTA, 2, cv2.LINE_AA)

    # lock-on box around the hand
    bx1, by1 = px.min(axis=0) - 22
    bx2, by2 = px.max(axis=0) + 22
    bx1, by1 = max(int(bx1), 2), max(int(by1), 2)
    bx2, by2 = min(int(bx2), VW - 3), min(int(by2), VIEW_H - 3)
    brackets(view, bx1, by1, bx2, by2, status_col, 18, 2)
    tag = f"ID::{shown}  {conf * 100:.0f}%" if locked else status
    put(view, tag, (bx1, max(by1 - 8, 18)), 0.5, status_col, 1)

    # top bar
    bar = view[0:34, :]
    view[0:34, :] = cv2.addWeighted(np.zeros_like(bar), 0.55, bar, 0.45, 0)
    put(view, TITLE, (12, 23), 0.55, CYAN, 1)
    put(view, time.strftime("%H:%M:%S"), (VW - 95, 23), 0.5, DIM, 1)

    # boot message in the first 1.6 seconds
    boot = time.time() - start
    if boot < 1.6:
        put(view, "INITIALIZING NEURAL CORE...", (VW // 2 - 150, VIEW_H // 2), 0.7, CYAN, 2)
        cv2.rectangle(view, (VW // 2 - 150, VIEW_H // 2 + 16), (VW // 2 + 150, VIEW_H // 2 + 28), CYAN, 1)
        cv2.rectangle(view, (VW // 2 - 150, VIEW_H // 2 + 16),
                      (VW // 2 - 150 + int(300 * boot / 1.6), VIEW_H // 2 + 28), CYAN, -1)

    # ---- right-hand panel ----
    if panel_bg is None:
        panel_bg = make_panel_background(VIEW_H)
    panel = panel_bg.copy()

    put(panel, "// SYSTEM STATUS", (16, 28), 0.45, DIM)
    if int(t * 2) % 2 == 0:
        cv2.circle(panel, (22, 54), 6, status_col, -1, cv2.LINE_AA)
    else:
        cv2.circle(panel, (22, 54), 6, status_col, 1, cv2.LINE_AA)
    put(panel, status, (38, 60), 0.62, status_col, 2)

    put(panel, "// DETECTED GESTURE", (16, 108), 0.45, DIM)
    if shown != last_text:
        last_text, typed = shown, 0.0
    typed = min(len(shown), typed + 0.5)
    text = shown[:int(typed)] + ("_" if int(t * 3) % 2 == 0 else " ")
    big_col = CYAN if locked else dim_colour(status_col, 0.8)
    put(panel, text, (16, 158), 1.0 if len(shown) <= 9 else 0.8, big_col, 2)
    put(panel, f"CONFIDENCE {conf * 100:5.1f}%", (16, 190), 0.55, WHITE, 1)
    cv2.line(panel, (16, 208), (PANEL_W - 16, 208), dim_colour(CYAN, 0.5), 1)

    put(panel, "// CLASS PROBABILITIES", (16, 238), 0.45, DIM)
    y0 = 270
    for i, lab in enumerate(labels):
        y = y0 + i * 34
        col = CYAN if (i == top and locked) else DIM
        put(panel, lab.upper().replace("_", " "), (16, y), 0.45, col)
        bx, bw, bh = 150, PANEL_W - 150 - 64, 12
        cv2.rectangle(panel, (bx, y - bh), (bx + bw, y), (70, 56, 36), 1)
        fill = int((bw - 2) * float(avg[i]))
        if fill > 0:
            cv2.rectangle(panel, (bx + 1, y - bh + 1), (bx + 1 + fill, y - 1), col, -1)
        put(panel, f"{avg[i] * 100:3.0f}%", (bx + bw + 8, y), 0.42, col)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
    last = now
    cv2.line(panel, (16, VIEW_H - 100), (PANEL_W - 16, VIEW_H - 100), dim_colour(CYAN, 0.5), 1)
    put(panel, f"FPS      {fps:5.1f}", (16, VIEW_H - 72), 0.5, GREEN, 1)
    put(panel, f"LATENCY  {latency:5.0f} ms", (16, VIEW_H - 48), 0.5, GREEN, 1)
    put(panel, f"FRAME    {frame_no:06d}", (16, VIEW_H - 24), 0.5, DIM, 1)

    canvas = np.hstack([view, panel])
    cv2.imshow(WIN, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        name_png = BASE / f"scan_{int(time.time())}.png"
        cv2.imwrite(str(name_png), canvas)
        print("Saved", name_png.name)

camera.release()
cv2.destroyAllWindows()