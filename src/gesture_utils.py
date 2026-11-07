def find_gesture(examples, gesture_name):
    for example in examples:
        if example["label"] == gesture_name:
            return example

    return None