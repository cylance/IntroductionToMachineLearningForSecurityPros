# -*- coding: utf-8 -*-
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.sparse import lil_matrix, vstack
from string import digits, punctuation, whitespace, ascii_letters

character_classes = [
    ascii_letters,
    digits,
    punctuation,
    whitespace
]


def get_character_to_character_transitions(sentences):
    transitions = {}
    for sentence in sentences:
        characters = list(sentence)
        previous = None
        for character in characters:
            cc = None
            for i, c in enumerate(character_classes):
                if character in c:
                    cc = i
                    break
            combo = (previous, cc)
            if combo not in transitions:
                transitions[combo] = 0
            transitions[combo] += 1

            previous = cc

        combo = (previous, None)
        if combo not in transitions:
            transitions[combo] = 0
        transitions[combo] += 1

    return transitions


def vectorize_sentence(sentence):
    transitions = get_character_to_character_transitions([sentence])
    vector = lil_matrix((4, 4), dtype=int)
    for (first, second), count in transitions.items():
        if first is None or second is None:
            continue

        vector[first, second] = count

    return vector.reshape((1, 4 * 4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster SMS messages with Gaussian Mixture Models"
    )
    parser.add_argument('-n', '--n_components', default=1, type=int, help="Number of clusters to produce")
    parser.add_argument('-r', '--print-results', default=False, action='store_true', help="Print out results per sample")
    parser.add_argument('-c', '--covariance-type', default='full', choices=["full", "tied", "diag", "spherical"], help="Covariance type")
    parser.add_argument('dataset', type=str, help="Path to dataset to read")

    args = parser.parse_args()

    vectors = []
    labels = []
    sentences = []

    print_clusters = args.print_results
    n_components = args.n_components
    covariance_type = args.covariance_type

    with open(args.dataset, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            label = parts[0]
            sentence = " ".join(parts[1:])
            sentences.append(sentence)

            vectors.append(vectorize_sentence(sentence))
            labels.append(1 if label == "spam" else 0)

    vectors = vstack(vectors).toarray()

    labels = np.array(labels)

    nb = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type
    )

    nb.fit(vectors)

    test_predict = nb.predict(vectors)

    clusters = {}
    cluster_labels = {}
    for index, sentence in enumerate(sentences):
        cluster = test_predict[index]

        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(sentence)

        if cluster not in cluster_labels:
            cluster_labels[cluster] = [0, 0]
        cluster_labels[cluster][labels[index]] += 1

    # compute cluster metrics (including the labels)
    for key in sorted(clusters.keys()):
        total_samples = len(clusters[key])
        p_ham = 100.0 * float(cluster_labels[key][0]) / float(total_samples)
        p_spam = 100.0 * float(cluster_labels[key][1]) / float(total_samples)

        print "Cluster {0} - Total Samples: {1} - Percent Ham: {2} - Percent Spam: {3}".format(
            key,
            total_samples,
            p_ham,
            p_spam
        )

    if print_clusters:
        for key in sorted(clusters.keys()):
            for s in clusters[key]:
                print "Cluster", key, " - ", s
