from idpanel.training.vectorization import load_raw_feature_vectors
from idpanel.training.features import load_raw_features
from idpanel.labels import load_labels
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import warnings
import os
import pickle


if __name__ == "__main__" or True:
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="Train Decision Tree",
    )
    parser.add_argument("-c", "--criterion", choices=["gini", "entropy"], default="gini")
    parser.add_argument("-s", "--splitter", choices=["random", "best"], default="best")
    parser.add_argument("-m", "--max-features", choices=["auto", "sqrt", "log2"], default=None)
    parser.add_argument("-d", "--max-depth", type=int, default=None)
    parser.add_argument("-S", "--min-samples-split", type=int, default=2)
    parser.add_argument("-l", "--min-samples-leaf", type=int, default=1)
    parser.add_argument("-w", "--min-weight-fraction-leaf", type=float, default=0.0)
    parser.add_argument("-n", "--max-leaf-nodes", type=int, default=None)

    args = parser.parse_args()

    warnings.warn = lambda x, y: x

    label_indeces = load_labels()
    raw_features = load_raw_features()
    original_labels, names, vectors = load_raw_feature_vectors()
    # Convert to binary label of panel/not panel
    labels = [1 if l != "not_panel" else 0 for l in original_labels]

    vectors = np.array(vectors)
    print "Creating training and testing sets"
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, stratify=labels)
    print X_train.shape[0], "samples in training set,", len(set(list(y_train))), "labels in training set"
    print X_test.shape[0], "samples in training set,", len(set(list(y_test))), "labels in testing set"

    decision_trees = {}

    dt = DecisionTreeClassifier(
        criterion=args.criterion,
        splitter=args.splitter,
        max_features=args.max_features,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,

    )
    dt.fit(X_train, y_train)

    print "Features required: ", (dt.feature_importances_ != 0).sum()

    pred = dt.predict(X_test)
    pred_proba = dt.predict_proba(X_test)

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
    export_graphviz(
        dt,
        out_file="dt.dot",
        feature_names=["{1}: {0}".format(i[0], i[1]) for i in raw_features],
        filled=True,
        impurity=True,
        #leaves_parallel=True,
        rounded=True,
        class_names=["Not Bot Panel", "Bot Panel"]
    )
    os.system("dot -Tpng dt.dot -o tree.png")
    with open("bot_model.mdl", "w") as f:
        pickle.dump({"model": dt, "relevant_features": dt.feature_importances_ != 0}, f)
