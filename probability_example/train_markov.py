import pykov
from random import choice
import numpy as np
from sklearn.metrics import classification_report
import cPickle as pickle

# pip install --upgrade git+git://github.com/riccardoscalco/Pykov@master


def get_character_to_character_transitions(sentences):
    transitions = {}
    for sentence in sentences:
        characters = list(sentence)
        previous = None
        for character in characters:
            combo = (previous, character)
            if combo not in transitions:
                transitions[combo] = 0
            transitions[combo] += 1

            previous = character

        combo = (previous, None)
        if combo not in transitions:
            transitions[combo] = 0
        transitions[combo] += 1

    return transitions


if __name__ == "__main__":
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    with open("data/SMSSpamCollection", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split()
            label = parts[0]
            sentence = " ".join(parts[1:])

            # Not the best way to do this
            if choice([True, True, True, False]):
                X_train.append(sentence)
                y_train.append(1 if label == "spam" else 0)
            else:
                X_test.append(sentence)
                y_test.append(1 if label == "spam" else 0)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    ham_sentences = []
    spam_sentences = []

    for index in xrange(len(X_train)):
        if y_train[index] == 1:
            spam_sentences.append(X_train[index])
        else:
            ham_sentences.append(X_train[index])

    ham_transitions = get_character_to_character_transitions(ham_sentences)
    spam_transitions = get_character_to_character_transitions(spam_sentences)

    ham_chain = pykov.Chain(ham_transitions)
    spam_chain = pykov.Chain(spam_transitions)

    predictions = []

    for index in xrange(len(X_test)):
        h = ham_chain.walk_probability(list(X_test[index]))
        s = spam_chain.walk_probability(list(X_test[index]))
        pred = 1 if s > h else 0
        predictions.append(pred)

    predictions = np.array(predictions)

    print "Classification report for testing set"
    print "Accuracy:", (predictions == y_test).mean()
    print classification_report(y_test, predictions)

    with open("markov_model.pkl", "w") as f:
        pickle.dump({
            "ham": ham_transitions,
            "spam": spam_transitions
        }, f)
