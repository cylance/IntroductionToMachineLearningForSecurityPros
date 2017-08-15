# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import lil_matrix, vstack
from sklearn.metrics import classification_report
from string import digits, punctuation, whitespace, ascii_letters
import cPickle as pickle

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
        description="Train model to classify SMS as spam or ham"
    )
    parser.add_argument('-f', '--fit-prior', default=False, action="store_true", help="Learn class prior probabilities")
    parser.add_argument('-a', '--alpha', default=1.0, type=float, help="Smoothing Parameter")
    parser.add_argument('dataset', type=str, help="Path to dataset to read")

    args = parser.parse_args()

    vectors = []
    labels = []

    with open(args.dataset, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            label = parts[0]
            sentence = " ".join(parts[1:])

            vectors.append(vectorize_sentence(sentence))
            labels.append(1 if label == "spam" else 0)

    vectors = vstack(vectors).tocsr()
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels)

    nb = MultinomialNB(
        fit_prior=args.fit_prior,
        alpha=args.alpha
    )

    nb.fit(X_train, y_train)
    #print (np.e ** nb.feature_log_prob_).tolist()
    #print nb.class_count_
    #print np.e ** nb.class_log_prior_

    print "Predict(Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country the Algarve is in? Txt ansr to 82277. £1.50 SP:Tyrone)"
    print nb.predict_proba(vectorize_sentence("Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country the Algarve is in? Txt ansr to 82277. £1.50 SP:Tyrone"))

    print "Predict(What time you coming down later?)"
    print nb.predict_proba(vectorize_sentence("What time you coming down later?"))

    test_predict = nb.predict(X_test)
    print "Classification report for testing set"
    print "Accuracy:", (test_predict == y_test).mean()
    print classification_report(y_test, test_predict)

    with open("naive_bayes.pkl", "w") as f:
        pickle.dump(nb, f)
