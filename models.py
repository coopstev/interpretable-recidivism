import numpy as np

from sklearn.svm import  LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, utils
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.neural_network import MLPClassifier

from data_loader import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)

RACELESS = True
RACE_COLS = [ "African_American", "Asian", "Hispanic", "Native_American" ]

METRICS = [ "accuracy", 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity' ]

def calculate_fraction(numerator, denominator):
    if numerator == 0 : return 0
    else : return numerator / denominator


def eval_performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    p = tp + fn
    n = tn + fp
    if metric == "accuracy" : return calculate_fraction(tp + tn, p + n)
    elif metric == "f1-score" : return calculate_fraction(2*tp, 2*tp + fp + fn)
    elif metric == "precision" : return calculate_fraction(tp, tp + fp)
    elif metric == "sensitivity" : return calculate_fraction(tp, p)
    elif metric == "specificity" : return calculate_fraction(tn, n)
    else : exit(1)



def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of LinearSVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1, 0}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    scores = []
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        if metric == "auroc" :
            scores.append(metrics.roc_auc_score(y_test, clf.decision_function(X_test)))
        else : scores.append(eval_performance(y_test, clf.predict(X_test), metric))
    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True):
    """Search for hyperparameters of linear SVM with best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1, 0}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge")
        penalty: string specifying the penalty type used (default="l2")
        dual: boolean specifying whether to use the dual formulation of the
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    test_metrics = []
    for C_test in C_range :
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=C_test, random_state=445)
        test_metrics.append(cv_performance(clf, X, y, k, metric))
    best_test = np.argmax(test_metrics)
    return C_range[best_test]
        

def random_forest(features, X_train, y_train, X_test, y_test, m, n_clf=10):
    """
    Returns accuracy on the test set X_test with corresponding labels y_test
    using a random forest classifier with n_clf decision trees trained with
    training examples X_train and training labels y_train.
    Input:
        X_train : np.array (n_train, d) - array of training feature vectors
        y_train : np.array (n_train) - array of labels corresponding to X_train samples
        X_test : np.array (n_test,d) - array of testing feature vectors
        y_test : np.array (n_test) - array of labels corresponding to X_test samples
        m : int - number of features to consider when splitting
        n_clf : int - number of decision tree classifiers in the random forest, default is 10
    Returns:
        accuracy : float - accuracy of random forest classifier on X_test samples
    """
    class_to_idx = {}
    idx_to_class = []
    for classification in y_train :
        if classification not in class_to_idx :
            class_to_idx[classification] = len(class_to_idx)
            idx_to_class.append(classification)
    num_classes = len(class_to_idx)
    forest_pred_votes = np.zeros((y_test.shape[0], num_classes))
    for i in range(n_clf) :
        X_sample, y_sample = utils.resample(X_train, y_train, replace=True, n_samples=y_train.shape[0], stratify=y_train)
        clf = DecisionTreeClassifier(criterion='entropy', max_features=m)
        clf.fit(X_sample, y_sample)
        plot_tree(clf, max_depth=m, feature_names=features, filled=True, label='all', fontsize=12)
        plt.savefig("tree" + str(m) + "-" + str(i+1) + ".png")
        plt.close()
        tree_preds = clf.predict(X_test)
        for j in range(tree_preds.shape[0]) :
            pred_class = tree_preds[j]
            pred_idx = class_to_idx[pred_class]
            forest_pred_votes[j][pred_idx] += 1
    forest_preds = [ idx_to_class[np.argmax(forest_pred_votes[j])] for j in range(y_test.shape[0]) ]
    DecTree_findings = [ ["Performance Measures", "Performance"] ]
    for metric in METRICS :
        if metric == "auroc" :
            prob_of_1 = [ votes[class_to_idx[1]] / n_clf for votes in forest_pred_votes ]
            performance = metrics.roc_auc_score(y_test, prob_of_1)
        else : performance = eval_performance(y_test, forest_preds, metric)
        DecTree_findings.append([metric, performance])
    print(DecTree_findings)
    return DecTree_findings


def main():
    FEATURES, X_train, Y_train, X_test, Y_test = get_split_binary_data(
        fname="propublica_data_for_fairml.csv"
    )

    print("BEGIN FINDING OPTIMAL REGULARIZATION CONSTANTS FOR SVM")
    C_RANGE = [ 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 ]
    reg_const_findings = [ ["Performance Measures", 'C', "Performance"] ]
    for metric in METRICS :
        C_best = select_param_linear(X_train, Y_train, 5, metric, C_RANGE)
        clf = LinearSVC(loss='hinge', C=C_best, random_state=445)
        performance = cv_performance(clf, X_train, Y_train, 5, metric)
        reg_const_findings.append([metric, C_best, performance])
    print(reg_const_findings)

    print("BEGIN RETRIEVING SVM WEIGHTS")
    COLS = FEATURES.keys()
    for metric, C_best, performance in reg_const_findings[1:] :
        clf = LinearSVC(loss='hinge', C=C_best, random_state=445)
        clf.fit(X_train, Y_train)
        theta = clf.coef_[0]
        word_coeffs = [ (word, theta[FEATURES[word]]) for word in COLS ]
        word_coeffs.sort(key=lambda a: a[1])
        print("\n\nWhen optimizing for", metric, "we get a performance of", performance, "and the following coeffs:")
        for i in range(len(FEATURES)) :
            print(word_coeffs[-1*(i+1)][0], "is the number", i+1, "most positive with a coefficient of", word_coeffs[-1*(i+1)][1])
        print("And the bias is", clf.intercept_[0])
    
    print("BEGIN RANDOM FORESTS")
    m_vals = list(range(1, len(COLS), 4))
    for m in m_vals:
        print('m = {}'.format(m))
        random_forest(list(COLS), X_train, Y_train, X_test, Y_test, m, n_clf=10)
    
    print("BEGIN NN")
    clf = MLPClassifier(hidden_layer_sizes = (100,100), random_state=445, max_iter=200, verbose=True, early_stopping=True).fit(X_train, Y_train)
    pred_probs = clf.predict_proba(X_test)
    preds = [ np.argmax(probs) for probs in pred_probs ]
    NN_findings = [ ["Performance Measures", "Performance"] ]
    for metric in METRICS :
        if metric == "auroc" :
            performance = metrics.roc_auc_score(Y_test, pred_probs.transpose()[1])
        else : performance = eval_performance(Y_test, preds, metric)
        NN_findings.append([metric, performance])
    print(NN_findings)


if __name__ == "__main__":
    main()
