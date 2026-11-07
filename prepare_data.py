from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

DATASET_DIR = Path("data/data_set/leapGestRecog")

IMAGE_SIZE = (60, 160)   # (height, width) - keeps the dataset's wide shape
BATCH_SIZE = 32
EPOCHS = 30

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Gesture folders to leave out completely (not used for train/val/test)
EXCLUDE_CLASSES = {
    "04_fist_moved",
    "08_palm_moved",
    "10_down",
    "02_l",
    "07_ok",
    "03_fist",
}

TRAIN_PEOPLE = {"00", "01", "02", "03", "04", "05", "06"}
VAL_PEOPLE = {"07"}
# people 08 and 09 are the test set (never seen during training)


# --------------------------------------------------
# 2. Find gesture names and create numeric labels
# --------------------------------------------------

people = sorted(
    folder
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir() and folder.name.isdigit()
)

if len(people) != 10:
    raise ValueError(f"Expected 10 person folders, found {len(people)}.")

class_names = sorted(
    folder.name
    for folder in people[0].iterdir()
    if folder.is_dir() and folder.name not in EXCLUDE_CLASSES
)

class_to_label = {name: label for label, name in enumerate(class_names)}

print("\nGesture labels:")
for name, label in class_to_label.items():
    print(f"{label}: {name}")

# Save class names so the webcam script can use the exact same order
Path("class_names.txt").write_text("\n".join(class_names), encoding="utf-8")


# --------------------------------------------------
# 3. Split images by person
# --------------------------------------------------

train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for person_folder in people:
    person_id = person_folder.name

    if person_id in TRAIN_PEOPLE:
        split = "train"
    elif person_id in VAL_PEOPLE:
        split = "validation"
    else:
        split = "test"

    for gesture_folder in sorted(person_folder.iterdir()):
        if not gesture_folder.is_dir():
            continue

        gesture_name = gesture_folder.name
        if gesture_name not in class_to_label:
            continue

        label = class_to_label[gesture_name]

        for image_path in sorted(gesture_folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if split == "train":
                train_paths.append(str(image_path))
                train_labels.append(label)
            elif split == "validation":
                val_paths.append(str(image_path))
                val_labels.append(label)
            else:
                test_paths.append(str(image_path))
                test_labels.append(label)

print("\nDataset summary:")
print(f"Training images:   {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Testing images:    {len(test_paths)}")


# --------------------------------------------------
# 4. Load and prepare images (grayscale, 60x160)
# --------------------------------------------------

def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image.set_shape([None, None, 1])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0

    return image, label


# --------------------------------------------------
# 5. Create datasets
# --------------------------------------------------

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Load + resize each image once, then keep it in memory (RAM)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    # Shuffle AFTER cache so the order still changes every epoch
    if shuffle:
        ds = ds.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)


# --------------------------------------------------
# 6. Build the model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),

    # Light augmentation (only active during training)
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.2),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 7. Train and monitor validation performance
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_gesture_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# 8. Save the model (best weights are restored)
# --------------------------------------------------

model.save("gesture_model.keras")
print("\nModel saved as gesture_model.keras")


# --------------------------------------------------
# 9. Final test on people 08-09
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%")


# --------------------------------------------------
# 10. Check which gestures the model confuses
# --------------------------------------------------

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\nTest results by actual gesture:")
print("Each row shows what the model predicted for that gesture.")

for i, gesture in enumerate(class_names):
    total = matrix[i].sum()
    correct = matrix[i, i]
    accuracy = (correct / total * 100) if total else 0

    print(f"\n{gesture} - correct: {correct}/{total} ({accuracy:.1f}%)")
    print("Predicted as:")

    for j, predicted_gesture in enumerate(class_names):
        count = matrix[i, j]
        if count:
            print(f"  {predicted_gesture}: {count}")