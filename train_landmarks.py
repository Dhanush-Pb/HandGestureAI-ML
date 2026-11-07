import json
import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --------------------------------------------------
# Settings
# --------------------------------------------------
QUICK_TEST = False       # True = 20-30 min test, False = medium run

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "data_set"
IMAGE_DIR = DATA_DIR / "freihand" / "rgb"

SRC = 224                # FreiHAND image size
IMG_SIZE = 192           # model input size
HEAT = IMG_SIZE // 4     # heat map size (48)
SIGMA = 2.0              # size of the Gaussian blob, in heat map cells
BATCH = 32

if QUICK_TEST:
    EPOCHS, STEPS = 3, 60
else:
    EPOCHS, STEPS = 12, 400      # 12 x 400 x 32 = 153,600 samples

PEAK_LR, MIN_LR, WARM = 5e-4, 2e-5, 2

MODEL_PATH = BASE / "landmark_heatmap_v3.keras"       # used by live_landmarks_v3.py
WEIGHTS_PATH = BASE / "best_v3.weights.h5"
BACKUP_DIR = BASE / "backup_v3"

# --------------------------------------------------
# 1. Labels: 3D points -> 2D pixels (in the 224 x 224 image)
# --------------------------------------------------
with open(DATA_DIR / "training_xyz.json") as f:
    xyz_all = np.array(json.load(f), dtype=np.float32)
with open(DATA_DIR / "training_K.json") as f:
    k_all = np.array(json.load(f), dtype=np.float32)
n_hands = len(xyz_all)


def project(xyz, K):
    uv = (K @ xyz.T).T
    return uv[:, :2] / uv[:, 2:3]


P_all = np.stack([project(xyz_all[a], k_all[a]) for a in range(n_hands)]).astype(np.float32)

image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
n_images = len(image_paths)
print("Images on disk:", n_images)

hand_of = np.arange(n_images) % n_hands
is_val = (hand_of % 10 == 0)                  # split by hand, not by image
train_idx = np.where(~is_val)[0]
val_pool = np.where(is_val)[0]
val_idx = np.random.default_rng(0).choice(val_pool, size=min(1500, len(val_pool)), replace=False)
print("Train images:", len(train_idx), " Validation images:", len(val_idx))

# --------------------------------------------------
# 2. One sample = image + target heat maps + true points
# --------------------------------------------------
def load_image(i):
    img = cv2.imread(str(image_paths[i]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_heat(pn):
    """pn: (21, 2) points in 0..1 -> (HEAT, HEAT, 21) Gaussian blobs, each summing to 1."""
    xs = (np.arange(HEAT) + 0.5)[None, None, :]
    ys = (np.arange(HEAT) + 0.5)[None, :, None]
    px = (pn[:, 0] * HEAT)[:, None, None]
    py = (pn[:, 1] * HEAT)[:, None, None]
    g = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * SIGMA ** 2))
    g /= g.sum(axis=(1, 2), keepdims=True) + 1e-8
    return np.transpose(g, (1, 2, 0)).astype(np.float32)


def augment(img, p, rng):
    f = IMG_SIZE / SRC
    # default (used only if no random placement keeps the whole hand inside)
    out = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    p_out = p * f

    for _ in range(10):
        pts = p.copy()
        src = img
        if rng.random() < 0.5:                       # mirror
            src = img[:, ::-1]
            pts[:, 0] = SRC - pts[:, 0]

        angle = rng.uniform(-45, 45)
        scale = rng.uniform(0.45, 1.2)               # < 1 = smaller hand, like a webcam
        M = cv2.getRotationMatrix2D((SRC / 2, SRC / 2), angle, scale)
        M[:, 2] += rng.uniform(-0.15, 0.15, 2) * SRC
        M = M * f
        p2 = pts @ M[:, :2].T + M[:, 2]

        if p2.min() >= 2 and p2.max() <= IMG_SIZE - 2:
            border = tuple(float(v) for v in rng.uniform(60, 200, 3))
            out = cv2.warpAffine(np.ascontiguousarray(src), M, (IMG_SIZE, IMG_SIZE),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border)
            p_out = p2
            break

    out = out.astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)      # contrast, brightness
    out = out * rng.uniform(0.9, 1.1, size=3)                     # colour cast
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5))
    out = out + rng.normal(0, rng.uniform(0, 6), out.shape)       # camera noise
    out = np.clip(out, 0, 255).astype(np.float32)
    return out, p_out / IMG_SIZE


def sample(i, rng, aug):
    img = load_image(i)
    p = P_all[hand_of[i]]
    if aug:
        out, pn = augment(img, p, rng)
    else:
        out = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        pn = p / SRC
    pn = np.clip(pn, 0.01, 0.99).astype(np.float32)
    return out, {"heat": make_heat(pn), "coords": pn.reshape(-1)}


signature = (
    tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
    {"heat": tf.TensorSpec((HEAT, HEAT, 21), tf.float32),
     "coords": tf.TensorSpec((42,), tf.float32)},
)


def make_ds(indices, aug, seed, infinite):
    def gen():
        rng = np.random.default_rng(seed)
        while True:
            for i in rng.permutation(indices):
                yield sample(int(i), rng, aug)
            if not infinite:
                return
    ds = tf.data.Dataset.from_generator(gen, output_signature=signature)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_idx, aug=True, seed=1, infinite=True)
val_ds = make_ds(val_idx, aug=False, seed=2, infinite=False)
hard_val_ds = make_ds(val_idx[:600], aug=True, seed=3, infinite=False)

# --------------------------------------------------
# 3. Model: MobileNetV2 + U-Net style head -> 21 heat maps
# --------------------------------------------------
class SoftArgmax(layers.Layer):
    """21 heat maps -> 21 (x, y) points between 0 and 1."""

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


backbone = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

# three feature maps: 24x24 (fine detail), 12x12, 6x6 (big picture)
features = keras.Model(backbone.input, [
    backbone.get_layer("block_6_expand_relu").output,
    backbone.get_layer("block_13_expand_relu").output,
    backbone.output,
])

inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)
c3, c4, c5 = features(x, training=False)       # BatchNorm stays in stable mode


def up(x, skip=None, ch=128):
    x = layers.Conv2DTranspose(ch, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=16)(x)
    if skip is not None:
        s = layers.Conv2D(ch, 1, use_bias=False)(skip)
        s = layers.GroupNormalization(groups=16)(s)
        x = layers.Add()([x, s])
    return layers.ReLU()(x)


x = up(c5, c4)       # 6  -> 12
x = up(x, c3)        # 12 -> 24
x = up(x)            # 24 -> 48
heat = layers.Conv2D(21, 1, name="heat")(x)
coords = SoftArgmax(name="coords")(heat)

model = keras.Model(inputs, {"heat": heat, "coords": coords})
heat_model = keras.Model(inputs, heat)         # this one is saved for the live test


def heat_loss(y_true, y_pred):
    """Cross-entropy between the predicted map and the Gaussian target."""
    k = y_pred.shape[-1]
    logits = tf.reshape(y_pred, (-1, HEAT * HEAT, k))
    target = tf.reshape(y_true, (-1, HEAT * HEAT, k))
    logp = tf.nn.log_softmax(logits, axis=1)
    return tf.reduce_mean(-tf.reduce_sum(target * logp, axis=1), axis=-1)


model.compile(
    optimizer=keras.optimizers.Adam(PEAK_LR),
    loss={"heat": heat_loss, "coords": keras.losses.MeanAbsoluteError()},
    loss_weights={"heat": 1.0, "coords": 10.0},
    metrics={"coords": ["mae"]},
)


def lr_at(epoch, lr=None):
    if epoch < WARM:
        return PEAK_LR * (epoch + 1) / (WARM + 1)
    t = (epoch - WARM) / max(1, EPOCHS - WARM)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * t))


callbacks = [
    keras.callbacks.LearningRateScheduler(lr_at, verbose=1),
    keras.callbacks.ModelCheckpoint(str(WEIGHTS_PATH), monitor="val_loss",
                                    save_best_only=True, save_weights_only=True, verbose=1),
    keras.callbacks.BackupAndRestore(str(BACKUP_DIR)),   # resume after a restart
]

# --------------------------------------------------
# 4. Train
# --------------------------------------------------
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          steps_per_epoch=STEPS, callbacks=callbacks)

# --------------------------------------------------
# 5. Save best model, measure error, save previews
# --------------------------------------------------
model.load_weights(WEIGHTS_PATH)
heat_model.save(MODEL_PATH)


def eval_px(ds):
    errs = []
    for xb, yb in ds:
        pred = model(xb, training=False)["coords"].numpy()
        errs.append(np.abs(pred - yb["coords"].numpy()).reshape(-1))
    return np.concatenate(errs).mean() * 224


print(f"\nValidation error, normal images:    about {eval_px(val_ds):.1f} px (224x224 image)")
print(f"Validation error, rotated / zoomed: about {eval_px(hard_val_ds):.1f} px")
print("Saved", MODEL_PATH.name)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def save_preview(aug, path):
    rng = np.random.default_rng(5)
    images, truth = [], []
    for i in val_idx[:8]:
        img, lab = sample(int(i), rng, aug)
        images.append(img)
        truth.append(lab["coords"])
    pred = model(np.array(images, dtype=np.float32), training=False)["coords"].numpy()
    zoom, tiles = 3, []
    for img, p, t in zip(images, pred, truth):
        tile = cv2.cvtColor(cv2.resize(img.astype(np.uint8), None, fx=zoom, fy=zoom),
                            cv2.COLOR_RGB2BGR)
        size = IMG_SIZE * zoom
        for pts, color in ((t, (0, 255, 0)), (p, (0, 0, 255))):   # green = truth, red = model
            xy = (pts.reshape(21, 2) * size).astype(int)
            for a_, b_ in CONNECTIONS:
                cv2.line(tile, tuple(xy[a_]), tuple(xy[b_]), color, 1)
            for pt in xy:
                cv2.circle(tile, tuple(pt), 3, color, -1)
        tiles.append(tile)
    rows = [np.hstack(tiles[r:r + 4]) for r in range(0, 8, 4)]
    cv2.imwrite(str(path), np.vstack(rows))


save_preview(False, BASE / "v3_preview.png")
save_preview(True, BASE / "v3_preview_hard.png") 
print("Saved v3_preview.png and v3_preview_hard.png")
