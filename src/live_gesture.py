
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from landmark_utils import (
    load_landmark_model,
    to_model_input,
    decode,
    guide_box,
    normalize,
    OFFSET,
    SCALE,
    CONNECTIONS,
)

from common.paths import (
    PROJECT_ROOT,
    GESTURE_MODEL,
    GESTURE_LABELS,
)

BASE = PROJECT_ROOT

# ---------------- settings ----------------
SKIP = 2
SMOOTH = 6

# When is a gesture accepted?
CONFIDENCE = 0.60
MIN_CONF = 0.40
MARGIN = 0.15
CONFIRM = 2
HOLD_SECONDS = 0.6

# Name shown on screen for each gesture
DISPLAY = {
    "five": "open hand",
    "flat_hand": "flat hand",
}

SHOW_PERCENT = True
SHOW_SECOND = True

TITLE = "HAND GESTURE RECOGNITION"

# ---------------- colours (BGR) ----------------
CYAN = (255, 255, 0)
LOCK = (0, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = 2


def put(img, text, org, scale=0.6):
    cv2.putText(img, text, org, FONT, scale, WHITE, BOLD, cv2.LINE_AA)


def put_right(img, text, y, scale=0.5):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, BOLD)
    put(img, text, (img.shape[1] - tw - 12, y), scale)


def put_center(img, text, cx, y, scale=0.5):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, BOLD)
    put(img, text, (cx - tw // 2, y), scale)


def dark_strip(img, y1, y2, alpha=0.55):
    """Dark see-through strip so the white text is easy to read."""
    part = img[y1:y2]
    img[y1:y2] = cv2.addWeighted(
        np.zeros_like(part), alpha, part, 1 - alpha, 0
    )


def corner_box(img, x1, y1, x2, y2, color, length=28, thick=3):
    """Guide box drawn as a thin frame with strong corners."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    for px, py, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(
            img, (px, py), (px + dx * length, py),
            color, thick, cv2.LINE_AA
        )
        cv2.line(
            img, (px, py), (px, py + dy * length),
            color, thick, cv2.LINE_AA
        )


# ---------------- models ----------------
landmarks = load_landmark_model()
fast = tf.function(lambda x: landmarks(x, training=False))

classifier = keras.models.load_model(GESTURE_MODEL)

with open(GESTURE_LABELS, "r") as f:
    labels = json.load(f)

print("Gestures:", labels)

names = [
    DISPLAY.get(lab, lab.replace("_", " "))
    for lab in labels
]


def infer(frame):
    """Hand points and gesture probabilities for one camera frame."""
    x1, y1, side = guide_box(frame)

    crop = frame[y1:y1 + side, x1:x1 + side]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    heat = fast(
        tf.constant(to_model_input(rgb)[np.newaxis])
    ).numpy()[0]

    pts = (decode(heat) - OFFSET) / SCALE

    probs = classifier(
        normalize(pts)[np.newaxis],
        training=False
    ).numpy()[0]

    return pts, probs


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

print("Q = quit, S = save a screenshot")

# Warm-up: the first model call is slow.
ok, frame = camera.read()

if not ok:
    raise RuntimeError("Could not read a camera frame.")

pts, _ = infer(cv2.flip(frame, 1))
infer(cv2.flip(frame, 1))

recent = deque(maxlen=SMOOTH)
m = np.zeros(len(names))
ranked = np.arange(len(names))

shown, hold_until = None, 0.0
candidate, votes = None, 0

latency, fps, frame_no = 0.0, 0.0, 0
last = time.time()

while True:
    ok, frame = camera.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    x1, y1, side = guide_box(frame)

    # ---- run the models and decide ----
    if frame_no % SKIP == 0:
        t0 = time.time()

        pts, probs = infer(frame)

        recent.append(probs)
        m = np.mean(recent, axis=0)
        ranked = np.argsort(m)[::-1]

        top = int(ranked[0])
        conf = float(m[top])

        second_conf = (
            float(m[ranked[1]])
            if len(ranked) > 1 else 0.0
        )

        sure = (
            conf >= CONFIDENCE
            or (
                conf >= MIN_CONF
                and conf - second_conf >= MARGIN
            )
        )

        if sure:
            votes = votes + 1 if top == candidate else 1
            candidate = top

            if votes >= CONFIRM or shown == top:
                shown = top
                hold_until = time.time() + HOLD_SECONDS

        else:
            candidate, votes = None, 0

        ms = (time.time() - t0) * 1000
        latency = (
            ms if latency == 0
            else 0.8 * latency + 0.2 * ms
        )

    frame_no += 1

    active = shown is not None and time.time() < hold_until

    # ---- dark strips for the text ----
    dark_strip(frame, 0, 78)
    dark_strip(frame, H - 34, H)

    # ---- guide box and hand skeleton ----
    corner_box(
        frame,
        x1,
        y1,
        x1 + side,
        y1 + side,
        LOCK if active else CYAN
    )

    px = (pts * side + np.array([x1, y1])).astype(int)

    for a, b in CONNECTIONS:
        cv2.line(
            frame,
            tuple(px[a]),
            tuple(px[b]),
            GREEN,
            2
        )

    for p in px:
        cv2.circle(frame, tuple(p), 4, RED, -1)

    # ---- top left: gesture name and second line ----
    if active:
        label = names[shown] + (
            f" ({m[shown] * 100:.0f}%)"
            if SHOW_PERCENT else ""
        )

        others = [
            int(i) for i in ranked
            if int(i) != shown
        ]

        second_text = None

        if others:
            second_text = f"2nd: {names[others[0]]}"

            if SHOW_PERCENT:
                second_text += f" ({m[others[0]] * 100:.0f}%)"

    else:
        label = "no gesture"

        best = int(ranked[0])
        second_text = f"best guess: {names[best]}"

        if SHOW_PERCENT:
            second_text += f" ({m[best] * 100:.0f}%)"

    put(frame, label, (12, 38), 1.0)

    if SHOW_SECOND and second_text:
        put(frame, second_text, (12, 66), 0.55)

    # ---- title, hint, speed and keys ----
    put_right(frame, TITLE, 26, 0.55)

    put_center(
        frame,
        "Keep your hand inside the box",
        x1 + side // 2,
        min(y1 + side + 26, H - 44),
        0.55
    )

    now = time.time()

    inst = 1 / max(now - last, 1e-6)
    fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
    last = now

    put(frame, f"{fps:.0f} FPS  |  {latency:.0f} ms", (12, H - 11), 0.55)
    put_right(frame, "Q quit   S screenshot", H - 11, 0.5)

    cv2.imshow("Gesture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        shot = BASE / f"shot_{int(time.time())}.png"
        cv2.imwrite(str(shot), frame)
        print("Saved", shot.name)


camera.release()
cv2.destroyAllWindows()
