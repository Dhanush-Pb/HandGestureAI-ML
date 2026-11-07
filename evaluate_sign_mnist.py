from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent

TEST_CSV = PROJECT_DIR / "data" / "sign_mnist_test.csv"
MODEL_PATH = PROJECT_DIR / "sign_mnist_model.keras"


# --------------------------------------------------
# 2. Load the test data
# --------------------------------------------------

print("Loading test data...")

test_data = pd.read_csv(TEST_CSV)

labels = test_data.iloc[:, 0].to_numpy()
pixels = test_data.iloc[:, 1:].to_numpy()

# Convert 784 pixels into 28 x 28 images
images = pixels.reshape(-1, 28, 28, 1).copy()

# Normalize pixel values from 0–255 to 0–1
images = images / 255.0

print(f"Test images: {len(images)}")


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# 4. Make predictions
# --------------------------------------------------

print("Predicting test images...")

probabilities = model.predict(images, verbose=1)

# Select the class with the highest probability
predictions = np.argmax(probabilities, axis=1)


# --------------------------------------------------
# 5. Print accuracy
# --------------------------------------------------

accuracy = accuracy_score(labels, predictions)

print("\n------------------------------")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("------------------------------")


# --------------------------------------------------
# 6. Display the confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    labels,
    predictions,
    labels=list(range(25))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=list(range(25))
)

display.plot(
    cmap="Blues",
    xticks_rotation="vertical",
    values_format="d"
)

plt.title("Sign Language MNIST - Confusion Matrix")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Show examples the model got wrong
# --------------------------------------------------

wrong_indices = np.where(predictions != labels)[0]

print(f"\nIncorrect predictions: {len(wrong_indices)}")

if len(wrong_indices) == 0:
    print("The model predicted every test image correctly!")
else:
    # Show up to 12 incorrect examples
    number_to_show = min(12, len(wrong_indices))

    plt.figure(figsize=(12, 8))

    for plot_index in range(number_to_show):
        image_index = wrong_indices[plot_index]

        true_label = labels[image_index]
        predicted_label = predictions[image_index]

        plt.subplot(3, 4, plot_index + 1)

        plt.imshow(
            images[image_index].squeeze(),
            cmap="gray"
        )

        plt.title(
            f"Actual: {true_label}\nPredicted: {predicted_label}"
        )

        plt.axis("off")

    plt.suptitle("Examples of Incorrect Predictions")
    plt.tight_layout()
    plt.show()