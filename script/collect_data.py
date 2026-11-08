import cv2
from pathlib import Path

# Save images in data/raw/open_hand
save_folder = Path("data/raw/open_hand")
save_folder.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)
image_count = len(list(save_folder.glob("*.jpg")))

if not camera.isOpened():
    print("Could not open camera.")
    raise SystemExit

print("Show an open hand to the camera.")
print("Press S to save an image. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"Open hand images: {image_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Collect Open Hand Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        image_count += 1
        image_path = save_folder / f"open_hand_{image_count:04d}.jpg"
        cv2.imwrite(str(image_path), frame)
        print(f"Saved: {image_path}")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()