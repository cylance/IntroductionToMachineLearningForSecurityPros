import json


def load_labels():
    with open("labels.json", "r") as f:
        return json.load(f)