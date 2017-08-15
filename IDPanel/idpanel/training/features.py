import json


def reduce_prevector_datapoints_to_features(data_points):
    offsets = {}
    for point in data_points:
        if point['offset'] not in offsets:
            offsets[point['offset']] = set()
        offsets[point['offset']].add((point['offset'], point['code'], point['content_ssdeep']))

    features = []
    for offset in offsets.keys():
        for point in offsets[offset]:
            features.append(point)

    return features


def load_raw_features():
    with open("raw_features.json", "r") as f:
        return json.load(f)
