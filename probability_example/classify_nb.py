from scipy.sparse import lil_matrix, vstack
import cPickle as pickle
import sys
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
    if len(sys.argv) == 1:
        print "python classify_model.py 'string to classify'"
        sys.exit(0)

    to_classify = " ".join(sys.argv[1:])

    with open("naive_bayes.pkl", "r") as f:
        classifier = pickle.load(f)

    vector = vectorize_sentence(to_classify)
    prediction = classifier.predict(vector)
    proba = classifier.predict_proba(vector)

    print "[ham] ", proba[0][0]
    print "[spam] ", proba[0][1]

    print "Prediction:", "spam" if prediction == 1 else "ham"
