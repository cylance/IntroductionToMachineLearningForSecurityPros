import math
import random
import numpy as np


class DecisionTree:

    def __init__(self, allowed_feature_indeces, features_to_choose_from=20):
        self.tree = None
        self.allowed_feature_indeces = allowed_feature_indeces
        self.features_used = []
        self.top_features = features_to_choose_from

    def _calculate_entropy(self, f_vals):
        labels = set([i[0] for i in f_vals])
        t_vals = len(f_vals)
        entropy = 0.0
        for label in labels:
            label_counts = len([i[0] for i in f_vals if i[0] == label])
            pcis = float(label_counts) / float(t_vals)
            entropy += (-1 * pcis * math.log(pcis, 2))

        return entropy

    def _find_optimal_split(self, vectors, feature_index, labels):
        # First we determine the split points
        f_vals = sorted([(labels[i[0]], i[1]) for i in enumerate(list(vectors[:, feature_index]))], key=lambda x: x[1])
        split_points = set()
        min_t = None
        best_split = None

        for f_index in xrange(1, len(f_vals)):
            if f_vals[f_index][0] != f_vals[f_index - 1][0]:
                split_point = float(f_vals[f_index][1] + f_vals[f_index - 1][1]) / 2.0
                if split_point not in split_points:
                    before = [i for i in f_vals if i[1] < split_point]
                    after = [i for i in f_vals if i[1] >= split_point]

                    # We don't weight these as our labels have very lopsided weights
                    t = self._calculate_entropy(before) + self._calculate_entropy(after)
                    if min_t is None or t < min_t:
                        min_t = t
                        best_split = split_point

                split_points.add(split_point)

        return best_split, min_t

    def fit(self, vectors, labels):
        self.features_used = set()
        self.tree = {}
        job_queue = [(vectors, labels, "")]

        while len(job_queue) != 0:
            v, l, path = job_queue.pop(0)
            if len(set(l)) == 1:
                # determine if we need to branch anymore
                self.tree[path] = {"labels": {l[0]: len(l)}}
                continue

            sorted_features = sorted([(feature_index, self._find_optimal_split(v, feature_index, l)) for feature_index in self.allowed_feature_indeces], key=lambda x: x[1][1])

            feature, split = random.choice(
                [(i[0], i[1][0]) for i in sorted_features[:self.top_features if len(sorted_features) > self.top_features else len(sorted_features)]]
            )
            self.features_used.add(feature)
            self.tree[path] = {"split": split, "feature": feature, "labels": {}}
            for label in set(l):
                self.tree[path]['labels'][label] = len([i for i in l if i == label])

            bv = []
            av = []
            bl = []
            al = []

            for index in xrange(v.shape[0]):
                if v[index, feature] < split:
                    bv.append(v[index, :])
                    bl.append(l[index])
                else:
                    av.append(v[index, :])
                    al.append(l[index])

            if len(bv) == 0 or len(av) == 0:
                # If we don't get any split, something is wrong...
                continue

            bv = np.vstack(bv)
            av = np.vstack(av)
            job_queue.append((av, al, path + "a"))
            job_queue.append((bv, bl, path + "b"))

        self.features_used = list(set(self.features_used))

    def _predict_vector(self, vector):
        if len(vector.shape) == 1:
            vector = vector.reshape(1, vector.shape[0])
        path = ""
        most_likely_label = None
        while path in self.tree:
            node = self.tree[path]

            total_samples = float(sum(node['labels'].values()))
            most_likely_label = {}
            for label in node['labels'].keys():
                most_likely_label[int(label)] = float(node['labels'][label]) / total_samples

            #most_likely_label = sorted(node['labels'].items(), key=lambda x: x[1], reverse=True)[0][0]
            if "split" in node:
                split = node["split"]
                feature = node["feature"]
                if vector[0, feature] < split:
                    path += "b"
                else:
                    path += "a"
            else:
                break
        return most_likely_label

    def predict_probs(self, vectors):
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, vectors.shape[0])
        results = []
        for index in xrange(vectors.shape[0]):
            results.append(self._predict_vector(vectors[index, :]))

        return results

    def predict(self, vectors):
        probs = self.predict_probs(vectors)
        results = []
        for index in xrange(vectors.shape[0]):
            results.append(sorted(probs[index].items(), key=lambda x: x[1], reverse=True)[0][0])

        return results

    def score(self, vectors, labels):
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, vectors.shape[0])
        results = self.predict(vectors)
        matches = len([ri for ri in xrange(len(labels)) if labels[ri] == results[ri]])
        return float(matches) / float(len(labels))


if __name__ == "__main__":
    dt = DecisionTree([])

    print 1 == dt._calculate_entropy(
        [
            (0, 0),
            (1, 0)
        ]
    )

    print 0 == dt._calculate_entropy(
        [
            (0, 0),
            (0, 0),
        ]
    )

