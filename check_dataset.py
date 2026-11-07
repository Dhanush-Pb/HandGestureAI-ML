from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = Path("data/data_set/leapGestRecog")

# Compare one training person with one testing person
people = ["00", "08"]

fig, axes = plt.subplots(2, 10, figsize=(18, 5))

for row, person in enumerate(people):
    person_folder = DATASET_DIR / person

    gesture_folders = sorted(
        folder for folder in person_folder.iterdir()
        if folder.is_dir()
    )

    for col, gesture_folder in enumerate(gesture_folders):
        images = list(gesture_folder.glob("*.png"))
        image_path = random.choice(images)

        image = Image.open(image_path)

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].set_title(gesture_folder.name)
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(
        "Training person 00" if person == "00" else "Test person 08"
    )

plt.tight_layout()
plt.show()