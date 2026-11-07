from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_DIR = Path("data/data_set/leapGestRecog")
IMAGE_SIZE = (64, 64)

# Load the trained model
model = tf.keras.models.load_model("gesture_model.keras")

# Keep gesture names in the same sorted order used during training
class_names = sorted(
    folder.name
    for folder in (DATASET_DIR / "00").iterdir()
    if folder.is_dir()
)

# Check 5 test images from the 06_index folder
image_folder = DATASET_DIR / "08" / "06_index"
image_paths = sorted(image_folder.glob("*.png"))[:5]

for image_path in image_paths:
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image) / 255.0
    image_array = tf.expand_dims(image_array, axis=0)

    scores = model.predict(image_array, verbose=0)[0]
    predicted_id = scores.argmax()
    confidence = scores[predicted_id] * 100

    print(f"\nImage: {image_path.name}")
    print(f"Actual gesture: 06_index")
    print(f"Predicted gesture: {class_names[predicted_id]}")
    print(f"Model confidence: {confidence:.1f}%")

    plt.imshow(image)
    plt.title(
        f"Actual: 06_index\nPredicted: {class_names[predicted_id]}"
    )
    plt.axis("off")
    plt.show()