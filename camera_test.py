from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "sign_mnist_model.keras"

CONFIDENCE_THRESHOLD = 70
SMOOTHING_FRAMES = 8
IMG_SIZE = 28

# Sign Language MNIST labels: J (9) and Z are not included.
# Label 9 is unused in the dataset.
LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J (not in dataset)",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]


# --------------------------------------------------
# Load model and camera
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera. Check that it is connected.")

recent_predictions = deque(maxlen=SMOOTHING_FRAMES)

print("Camera started. Place one hand inside the box.")
print("Press Q to quit.")


# --------------------------------------------------
# Camera loop
# --------------------------------------------------

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a camera frame.")
        break

    # Mirror the camera view
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    # Center square box for the hand
    box_size = min(int(height * 0.75), int(width * 0.75))
    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    hand_crop = frame[y1:y2, x1:x2]

    # Match the model's input: grayscale, 28x28, values from 0 to 1
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    input_image = resized.astype("float32") / 255.0
    input_image = input_image[np.newaxis, :, :, np.newaxis]

    # Predict and smooth the result
    probabilities = model.predict(input_image, verbose=0)[0]
    recent_predictions.append(probabilities)
    average = np.mean(recent_predictions, axis=0)

    predicted_index = int(np.argmax(average))
    confidence = float(average[predicted_index]) * 100

    if predicted_index == 9:
        result = "Unused class"
    else:
        result = LABELS[predicted_index]

    if confidence < CONFIDENCE_THRESHOLD:
        result = "Uncertain"

    color = (0, 255, 0) if result not in ("Uncertain", "Unused class") else (0, 0, 255)

    # Draw the prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{result}  {confidence:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("ASL Letter Camera Test", frame)
    cv2.imshow("Model input", cv2.resize(resized, (280, 280)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()