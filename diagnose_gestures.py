from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).parent
CSV = BASE / "data" / "gestures_hg14.csv"

df = pd.read_csv(CSV)
labels = sorted(df["label"].unique())
label_id = {l: i for i, l in enumerate(labels)}

X = df[[f"p{i}" for i in range(42)]].values.astype("float32")
y = df["label"].map(label_id).values

# Same train / validation split as train_gestures
is_val = np.zeros(len(df), dtype=bool)
for lab in labels:
    mask = df["label"] == lab
    takes = sorted(df.loc[mask, "take"].unique())
    if len(takes) >= 4:
        held = takes[3::4]
    elif len(takes) >= 2:
        held = [takes[-1]]
    else:
        held = []
    is_val |= (mask & df["take"].isin(held)).to_numpy()


def distance_features(X):
    """Distance between every pair of the 21 points (210 numbers).
    These do not change when the hand is rotated."""
    P = X.reshape(-1, 21, 2)
    i, j = np.triu_indices(21, k=1)
    return np.linalg.norm(P[:, i, :] - P[:, j, :], axis=2).astype("float32")


D = distance_features(X)

sets = {
    "raw 42 numbers (what your network uses)": X,
    "210 pair distances": D,
    "both together": np.hstack([X, D]),
}

print(f"Train rows: {(~is_val).sum()}   Validation rows: {is_val.sum()}\n")

for name, features in sets.items():
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    forest.fit(features[~is_val], y[~is_val])
    train_acc = forest.score(features[~is_val], y[~is_val])
    val_acc = forest.score(features[is_val], y[is_val])
    print(f"{name:42s} train {train_acc * 100:5.1f}%   validation {val_acc * 100:5.1f}%")