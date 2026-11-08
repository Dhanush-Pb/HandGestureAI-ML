
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
from common.paths import (
    GESTURE_CSV,
    GESTURE_MODEL,
    GESTURE_LABELS,
)

CSV = GESTURE_CSV
MODEL_OUT = GESTURE_MODEL
LABELS_OUT = GESTURE_LABELS
BATCH_SIZE = 32
EPOCHS = 150


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(CSV)

labels = sorted(df["label"].unique())

label_id = {
    label: i for i, label in enumerate(labels)
}

print("\nGestures:", labels)

print("\nImages per gesture:")
print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

X = df[
    [f"p{i}" for i in range(42)]
].values.astype("float32")

y = df["label"].map(label_id).values


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# Keep complete takes separate
# --------------------------------------------------

is_val = np.zeros(len(df), dtype=bool)

for lab in labels:

    class_mask = df["label"] == lab

    takes = sorted(
        df.loc[class_mask, "take"].unique()
    )

    if len(takes) >= 4:
        held = takes[3::4]

    elif len(takes) >= 2:
        held = [takes[-1]]

    else:
        held = []

    selected = (
        class_mask &
        df["take"].isin(held)
    )

    is_val |= selected.to_numpy()


# --------------------------------------------------
# VALIDATION DATA CHECK
# --------------------------------------------------

print("\nValidation samples per gesture:")

val_counts = (
    df.loc[is_val, "label"]
    .value_counts()
    .reindex(labels, fill_value=0)
)

print(val_counts.to_string())

if (val_counts == 0).any():
    raise ValueError(
        "Some gestures have no validation samples. "
        "Check the CSV take IDs."
    )

X_train = X[~is_val]
y_train = y[~is_val]

X_val = X[is_val]
y_val = y[is_val]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# --------------------------------------------------
# DATA AUGMENTATION
# Rotation, scaling, mirroring and noise
# --------------------------------------------------

def augment(X, y, copies, seed=0):

    rng = np.random.default_rng(seed)

    P = X.reshape(-1, 21, 2)

    Xs = [X]
    ys = [y]

    for _ in range(copies):

        # Random rotation
        ang = np.deg2rad(
            rng.uniform(-20, 20, len(P))
        )

        c = np.cos(ang)[:, None]
        s = np.sin(ang)[:, None]

        nx = (
            P[:, :, 0] * c -
            P[:, :, 1] * s
        )

        ny = (
            P[:, :, 0] * s +
            P[:, :, 1] * c
        )

        # Random horizontal mirroring
        flip = rng.random(len(P)) < 0.5

        nx[flip] *= -1

        # Random scaling
        Q = np.stack(
            [nx, ny],
            axis=2
        )

        Q = Q * rng.uniform(
            0.9,
            1.1,
            (len(P), 1, 1)
        )

        # Small random noise
        Q = Q + rng.normal(
            0,
            0.02,
            Q.shape
        )

        Xs.append(
            Q.reshape(
                len(P), 42
            ).astype("float32")
        )

        ys.append(y)

    return (
        np.concatenate(Xs),
        np.concatenate(ys)
    )


X_aug, y_aug = augment(
    X_train,
    y_train,
    copies=8
)

print("\nOriginal training samples:", len(X_train))
print("After augmentation:", len(X_aug))


# --------------------------------------------------
# BUILD NEURAL NETWORK
# --------------------------------------------------

model = keras.Sequential([

    layers.Input(shape=(42,)),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        len(labels),
        activation="softmax"
    )

])


# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nStarting gesture model training...\n")

history = model.fit(

    X_aug,
    y_aug,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=[
        early_stop,
        reduce_lr
    ],

    shuffle=True,

    verbose=2
)


# --------------------------------------------------
# SAVE MODEL AND LABELS
# --------------------------------------------------

model.save(MODEL_OUT)

with open(LABELS_OUT, "w") as f:
    json.dump(labels, f)


# --------------------------------------------------
# FINAL VALIDATION EVALUATION
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

pred = model.predict(
    X_val,
    verbose=0
).argmax(axis=1)


print("\n================================")
print("FINAL VALIDATION RESULTS")
print("================================")

print(
    f"Validation accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Validation loss: {loss:.4f}"
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nCLASSIFICATION REPORT:\n")

print(
    classification_report(
        y_val,
        pred,
        labels=range(len(labels)),
        target_names=labels,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nCONFUSION MATRIX")

print("Rows = true gesture")
print("Columns = predicted gesture")

print("\nGesture order:")
print(labels)

cm = confusion_matrix(
    y_val,
    pred,
    labels=range(len(labels))
)

print(cm)


# --------------------------------------------------
# COMPLETED
# --------------------------------------------------

print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print("Saved model:", MODEL_OUT.name)
print("Saved labels:", LABELS_OUT.name)
