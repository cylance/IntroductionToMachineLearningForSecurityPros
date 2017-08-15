from idpanel.training.vectorization import vectorize_with_sparse_features
from idpanel.decision_tree import DecisionTree
from json import load, dump


class ClassificationEngine:
    def __init__(self, decision_trees, sparse_features, total_feature_count, tree_per_label=-1):
        self.decision_trees = decision_trees
        self.tree_per_label = tree_per_label
        self.sparse_features = sparse_features
        self.total_feature_count = total_feature_count

    @staticmethod
    def load_model(file_path):
        with open(file_path, "r") as f:
            decision_trees, sparse_features, total_feature_count = load(f)

        dts = {}
        for key in decision_trees.keys():
            dts[key] = []
            for dt in decision_trees[key]:
                d = DecisionTree([])
                for k in dt['model'].keys():
                    setattr(d, k, dt['model'][k])
                dt['model'] = d
                dts[key].append(dt)

        return ClassificationEngine(
            decision_trees=dts,
            sparse_features=sparse_features,
            total_feature_count=total_feature_count
        )

    def save_model(self, file_path):
        with open(file_path, "w") as f:
            decision_trees = {}
            for key in self.decision_trees.keys():
                decision_trees[key] = []
                for dt in self.decision_trees[key]:
                    dt['model'].allowed_feature_indeces = []
                    dt['model'] = dt['model'].__dict__
                    decision_trees[key].append(dt)

            dump((decision_trees, self.sparse_features, self.total_feature_count), f)

    def get_required_requests(self):
        urls = set()

        for label in self.decision_trees.keys():
            requests_this_label = []
            for tree_index, tree in enumerate(self.decision_trees[label]):
                if self.tree_per_label > 0 and tree_index == self.tree_per_label:
                    break
                for features in tree["features"]:
                    requests_this_label.append(features[2][0])

            urls |= set(requests_this_label)

        return list(urls)

    def get_label_scores(self, c2_data, vector=None):
        if vector is None:
            vector = vectorize_with_sparse_features(self.sparse_features, self.total_feature_count, c2_data)
        label_results = {}

        best_choice = None
        best_score = 0

        for label in self.decision_trees.keys():
            label_results[label] = []
            for pair in self.decision_trees[label]:
                model = pair["model"]
                label_results[label].append(float(model.predict(vector)[0]))
            #print label_results[label]
            score = sum(label_results[label])
            if score > best_score:
                best_choice = label
                best_score = score

        return best_choice, label_results

    def get_label_probs(self, c2_data, vector=None):
        if vector is None:
            vector = vectorize_with_sparse_features(self.sparse_features, self.total_feature_count, c2_data)
        label_results = {}
        label_scores = {}

        best_choice = None
        best_score = 0

        for label in self.decision_trees.keys():
            label_results[label] = {1: 0.0, 0: 0.0}
            for pair in self.decision_trees[label]:
                model = pair["model"]
                probs = model.predict_probs(vector)[0]
                label_results[label][0] += 0.0 if 0 not in probs else probs[0]
                label_results[label][1] += 0.0 if 1 not in probs else probs[1]
            #print label_results[label]
            score = label_results[label][1] - label_results[label][0]
            score = float(score) / float(len(self.decision_trees[label]))
            label_scores[label] = score
            if score > best_score:
                best_choice = label
                best_score = score

        return best_choice, label_results, label_scores
