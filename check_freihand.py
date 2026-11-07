import json
import random
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Settings
# --------------------------------------------------

# Folder that holds training_xyz.json and training_K.json.
# Built from the script's own location, so it works from any terminal folder.
SEARCH_DIR = Path(__file__).parent / "data" / "data_set"
N_SAMPLES = 8       # how many images to draw
ZOOM = 2            # make the 224x224 images bigger

# Which of the 21 points connect (wrist = 0, then 4 points per finger)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]


# --------------------------------------------------
# 1. Find the label files and the image folder
# --------------------------------------------------

matches = list(SEARCH_DIR.rglob("training_xyz.json"))
if not matches:
    raise FileNotFoundError(
        f"Could not find training_xyz.json under '{SEARCH_DIR}'. "
        "Put training_xyz.json and training_K.json in that folder."
    )

root = matches[0].parent
print("Dataset folder:", root)

k_path = root / "training_K.json"
if not k_path.exists():
    raise FileNotFoundError(f"training_K.json is missing in {root}")

with open(root / "training_xyz.json") as f:
    xyz_all = np.array(json.load(f), dtype=np.float32)
with open(k_path) as f:
    k_all = np.array(json.load(f), dtype=np.float32)
print("Annotations:", xyz_all.shape, "(samples, points, xyz)")

# Try your folder name first, then the standard FreiHAND one
image_dir = None
for candidate in [root / "freihand" / "rgb", root / "training" / "rgb", root / "rgb"]:
    if candidate.is_dir():
        image_dir = candidate
        break

if image_dir is None:
    raise FileNotFoundError(
        f"Could not find an 'rgb' image folder inside {root}. "
        "Expected data_set\\freihand\\rgb"
    )

image_paths = sorted(image_dir.glob("*.jpg"))
print("Image folder:", image_dir)
print("Images found:", len(image_paths))

if not image_paths:
    raise FileNotFoundError(f"No .jpg images in {image_dir}")


# --------------------------------------------------
# 2. Turn the 3D points into 2D points on the image
# --------------------------------------------------

def project(xyz, K):
    """xyz: (21, 3) in camera space, K: (3, 3) camera matrix -> (21, 2) pixels."""
    uv = (K @ xyz.T).T
    return uv[:, :2] / uv[:, 2:3]


# --------------------------------------------------
# 3. Draw the 21 points on some random images
# --------------------------------------------------

random.seed(0)
chosen = random.sample(range(len(image_paths)), N_SAMPLES)
tiles = []

for i in chosen:
    image = cv2.imread(str(image_paths[i]))
    if image is None:
        print("Could not read", image_paths[i])
        continue
    image = cv2.resize(image, None, fx=ZOOM, fy=ZOOM)

    # The images come in 4 versions of the same 32,560 hands,
    # so one annotation belongs to 4 images.
    a = i % len(xyz_all)
    pts = project(xyz_all[a], k_all[a]) * ZOOM
    points = [(int(x), int(y)) for x, y in pts]

    for p, q in CONNECTIONS:
        cv2.line(image, points[p], points[q], (0, 255, 0), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

    cv2.putText(image, image_paths[i].name, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(image)

columns = 4
rows = [np.hstack(tiles[r:r + columns]) for r in range(0, len(tiles), columns)]
collage = np.vstack(rows)

cv2.imwrite("freihand_check.png", collage)
print("\nSaved freihand_check.png")

cv2.imshow("FreiHAND check - press any key to close", collage)
cv2.waitKey(0)
cv2.destroyAllWindows()