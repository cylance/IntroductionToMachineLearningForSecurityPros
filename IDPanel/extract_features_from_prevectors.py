import json
from idpanel.training.features import reduce_prevector_datapoints_to_features
from idpanel.blacklist import labels_to_ignore

if __name__ == "__main__":
    data_points = []
    with open("prevectors.json", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            line = json.loads(line)
            if line['label'] in labels_to_ignore:
                continue
            data_points.append(line)

    print "Loaded {0} prevectors".format(len(data_points))
    features = reduce_prevector_datapoints_to_features(data_points)
    print "Extracted {0} features".format(len(features))
    offsets = set([feature[0] for feature in features])
    print "Features cover {0} requests".format(len(offsets))
    with open("raw_features.json", "w") as f:
        json.dump(features, f)

    labels = sorted(set([dp["label"] for dp in data_points]))
    print "Vectors cover {0} labels".format(len(labels))
    with open("labels.json", "w") as f:
        json.dump(labels, f)
