from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).parent / "data" / "data_set" / "HG14-Hand Gesture"
TILE, PER_CLASS = 128, 6

classes = {}
for p in ROOT.rglob("*"):
    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        classes.setdefault(p.parent.name, []).append(p)

rows = []
for name in sorted(classes):
    files = sorted(classes[name])
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, PER_CLASS)]
    tiles = []
    for f in pick:
        img = cv2.resize(cv2.imread(str(f)), (TILE, TILE))
        tiles.append(img)
    row = np.hstack(tiles)
    cv2.putText(row, f"{name} ({len(files)})", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    rows.append(row)

cv2.imwrite(str(Path(__file__).parent / "hg14_preview.png"), np.vstack(rows))
print("Classes:", len(classes), "-> saved hg14_preview.png")