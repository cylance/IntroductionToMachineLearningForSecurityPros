from idpanel.training.vectorization import load_raw_feature_vectors
from idpanel.training.features import load_raw_features
from idpanel.labels import load_labels
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import pickle


def classify(model, sample):
    labels = sorted(model.keys())
    proba = []
    for label in labels:
        proba.append(model[label].predict_proba(sample)[0, 1])
    label = None
    proba = np.array(proba)
    if (proba > 0.5).sum() > 0:
        label = labels[proba.argmax()]
    return label, labels, proba


if __name__ == "__main__" or True:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="Train Logistic Regression Model",
    )
    parser.add_argument("-p", "--penalty", choices=["l1", "l2"], default="l2")
    parser.add_argument("-d", "--dual", action='store_true', default=False)
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument("-f", "--fit-intercept", default=True, action='store_true')
    parser.add_argument("-i", "--intercept-scaling", type=float, default=1.0)
    parser.add_argument("-m", "--max-iter", type=int, default=100)
    parser.add_argument("-s", "--solver", choices=["newton-cg", "lbfgs", "liblinear", "sag"], default="liblinear")
    parser.add_argument("-t", "--tol", type=float, default=0.0001)

    args = parser.parse_args()

    warnings.warn = lambda x, y: x

    label_indeces = load_labels()
    raw_features = load_raw_features()
    original_labels, names, vectors = load_raw_feature_vectors()
    labels = [1 if l != "not_panel" else 0 for l in original_labels]

    vectors = np.array(vectors)
    print "Creating training and testing sets"
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, stratify=labels)
    print X_train.shape[0], "samples in training set,", len(set(list(y_train))), "labels in training set"
    print X_test.shape[0], "samples in training set,", len(set(list(y_test))), "labels in testing set"

    lr = LogisticRegression(
        n_jobs=-1,
        penalty=args.penalty,
        dual=args.dual,
        C=args.C,
        fit_intercept=args.fit_intercept,
        intercept_scaling=args.intercept_scaling,
        max_iter=args.max_iter,
        solver=args.solver,
        tol=args.tol
    )
    lr.fit(X_train, y_train)

    #print (lr.feature_importances_ != 0).sum()

    pred = lr.predict(X_test)
    pred_proba = lr.predict_proba(X_test)

    print "Confusion Matrix:"
    print confusion_matrix(y_test, pred)

    #print np.array(y_test) == 1
    pos_hist, pos_bin_edges = np.histogram(pred_proba[np.array(y_test) == 1, 1],
                                           bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    neg_hist, neg_bin_edges = np.histogram(pred_proba[np.array(y_test) == 0, 1],
                                           bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    #print pos_hist.shape, pos_bin_edges.shape
    #print neg_hist.tolist()
    ax1.plot(pos_bin_edges[:-1] + 0.05, pos_hist, color='green', linestyle='solid', label="Positives")
    ax1.plot(neg_bin_edges[:-1] + 0.05, neg_hist, color='red', linestyle='solid', label="Negatives")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, max(neg_hist.max(), pos_hist.max()))
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Positive Classification Thresholds')
    ax1.legend(loc="upper left")

    fpr, tpr, _ = roc_curve(y_test, pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, linewidth=4)
    ax2.plot([0, 1], [0, 1], 'r--')
    #ax2.xlim([0.0, 1.0])
    #ax2.ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Logistic Regression ROC Curve')
    #ax2.legend(loc="lower right")
    plt.show()

    with open("bot_model.lrmdl", "w") as f:
        pickle.dump({"model": lr, "relevant_features": lr.coef_ != 0}, f)

