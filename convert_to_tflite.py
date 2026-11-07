import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from landmark_utils import (LANDMARK_MODEL_PATH, IMG_SIZE, INNER, PAD, OFFSET,
                            SCALE, decode, normalize, to_model_input)

BASE = Path(__file__).parent
OUT = BASE / "flutter_assets"
OUT.mkdir(exist_ok=True)
HG14 = BASE / "data" / "data_set" / "HG14-Hand Gesture"


class SoftArgmax(layers.Layer):
    """21 heat maps -> 42 numbers (x, y per point, 0..1). Same maths as decode()."""

    def call(self, heat):
        _, h, w, k = heat.shape
        flat = tf.reshape(heat, (-1, h * w, k))
        prob = tf.nn.softmax(flat, axis=1)
        xs = (tf.range(w, dtype=tf.float32) + 0.5) / w
        ys = (tf.range(h, dtype=tf.float32) + 0.5) / h
        gx = tf.reshape(tf.tile(xs[None, :], [h, 1]), (1, h * w, 1))
        gy = tf.reshape(tf.tile(ys[:, None], [1, w]), (1, h * w, 1))
        x = tf.reduce_sum(prob * gx, axis=1)
        y = tf.reduce_sum(prob * gy, axis=1)
        return tf.reshape(tf.stack([x, y], axis=-1), (-1, k * 2))


heat_model = keras.models.load_model(LANDMARK_MODEL_PATH)
inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
points_model = keras.Model(inp, SoftArgmax()(heat_model(inp, training=False)))
classifier = keras.models.load_model(BASE / "gesture_model.keras")


def convert(model, path, name):
    try:
        data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    except Exception as e:
        print(f"{name}: direct conversion failed ({type(e).__name__}), trying the SavedModel route")
        tmp = BASE / f"_{name}_saved"
        model.export(str(tmp))
        data = tf.lite.TFLiteConverter.from_saved_model(str(tmp)).convert()
        shutil.rmtree(tmp, ignore_errors=True)
    path.write_bytes(data)
    print(f"{name}: saved {path.name} ({len(data) / 1e6:.1f} MB)")


LM_PATH = OUT / "hand_landmarks.tflite"
CL_PATH = OUT / "gesture_classifier.tflite"
convert(points_model, LM_PATH, "landmarks")
convert(classifier, CL_PATH, "classifier")

labels = json.load(open(BASE / "gesture_labels.json"))
shutil.copy(BASE / "gesture_labels.json", OUT / "gesture_labels.json")
info = {"img_size": IMG_SIZE, "inner": INNER, "pad": PAD, "offset": OFFSET,
        "scale": SCALE, "guide_fraction": 0.8, "confidence": 0.7,
        "smooth_frames": 6, "labels": labels}
json.dump(info, open(OUT / "preprocess_info.json", "w"), indent=2)


# ---------- check: does the phone version give the same answers? ----------
def make_interp(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def run(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x.astype("float32"))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]


lm_it, cl_it = make_interp(LM_PATH), make_interp(CL_PATH)

files = []
for g in range(14):
    fs = sorted((HG14 / f"Gesture_{g}").glob("*.jpg"))
    files += [fs[i] for i in np.linspace(0, len(fs) - 1, 20).astype(int)]

point_diff, prob_diff, same = [], [], 0
for f in files:
    img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    canvas = to_model_input(img)[np.newaxis]

    pk = decode(heat_model(canvas, training=False).numpy()[0])
    pt = run(lm_it, canvas).reshape(21, 2)
    point_diff.append(np.abs(pk - pt).max())

    probs_k = classifier(normalize((pk - OFFSET) / SCALE)[np.newaxis], training=False).numpy()[0]
    probs_t = run(cl_it, normalize((pt - OFFSET) / SCALE)[np.newaxis])
    prob_diff.append(np.abs(probs_k - probs_t).max())
    same += int(probs_k.argmax() == probs_t.argmax())

print(f"\nChecked {len(files)} images")
print(f"Biggest point difference: {max(point_diff):.5f}  (fraction of the image)")
print(f"Biggest probability difference: {max(prob_diff):.4f}")
print(f"Same gesture from both versions: {same / len(files) * 100:.1f}%")
print("Files are in:", OUT)