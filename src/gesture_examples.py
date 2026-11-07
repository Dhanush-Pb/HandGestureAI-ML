# Finger order: thumb, index, middle, ring, pinky
# 1 = finger up, 0 = finger down
# These are simplified practice examples, not real camera data.
from gesture_utils import find_gesture

examples = [
    {"features": [1, 0, 0, 0, 0], "label": "thumbs_up"},
    {"features": [0, 0, 0, 0, 0], "label": "thumbs_down"},
    {"features": [0, 1, 1, 0, 0], "label": "peace_sign"},
    {"features": [1, 1, 1, 1, 1], "label": "open_hand"},
    {"features": [0, 0, 0, 0, 0], "label": "fist"},
    {"features": [0, 1, 0, 0, 0], "label": "pointing"},
    {"features": [1, 1, 0, 0, 0], "label": "ok_sign"},
    {"features": [0, 1, 1, 1, 0], "label": "three_fingers"},
    {"features": [0, 1, 1, 1, 1], "label": "four_fingers"},
    {"features": [0, 1, 0, 0, 1], "label": "rock_sign"},
    {"features": [1, 0, 0, 0, 1], "label": "call_me"},
    {"features": [1, 1, 1, 1, 1], "label": "stop"},
    {"features": [1, 1, 0, 0, 1], "label": "love_you"},
    {"features": [1, 1, 0, 0, 0], "label": "pinch"},
    {"features": [1, 1, 1, 1, 1], "label": "wave"}
]


# features = [example["features"] for example in examples]
# labels = [example["label"] for example in examples]

# print("Features:", features)
# print("Labels:", labels)

X = [example["features"] for example in examples]
y = [example["label"] for example in examples]

print("X:", X)
print("y:", y)
print("Number of examples:", len(X))
print("Number of labels:", len(y))

def show_examples():
   
    print("Total examples:", len(examples))

show_examples()

name = input("Enter a gesture name: ").strip().lower()
result = find_gesture(examples, name)

if result is not None:
    print("Found:", result)
else:
    print("Gesture not found.")


