"""EECS 445 - Fall 2022.

Project 1
"""

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from stack_data import RangeInLine


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    output_list = []
    word = ""
    input_string = input_string.lower()
    for i in string.punctuation:
        input_string = input_string.replace(i,' ')
    return input_string.split()
        


def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    num = 0
    for text in df['text']:
        output_list = extract_word(text)
        for word in output_list:
            if word not in word_dict:
                word_dict[word] = num
                num += 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    row = 0
    for text in df['text']:
        output_list = extract_word(text)
        for word in output_list:
            if word in word_dict:
                feature_matrix[row, word_dict[word]] = 1
        row += 1
    return feature_matrix


def performance(y_true, y_pred, metric="accuracy"):
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
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)

    m = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = m.ravel()
    if metric == "accuracy":
        return np.float64((tp+tn)/(tp+fn+fp+tn))
    if metric == "precision":
        return np.float64(tp/(tp+fp))
    if metric == "specificity":
        return np.float64(tn/(tn+fp))
    if metric == "f1-score": # 2*prec*sens/prec+sens
        prec = tp/(tp+fp)
        sens = tp/(tp+fn)
        return np.float64(2*prec*sens/(prec+sens))
    else: #metric == "sensitivity":
        return (np.float64(tp)/(tp+fn))


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # Put the performance of the model on each fold in the scores array
    scores = []
    strat = StratifiedKFold(n_splits=k)
    #strat.get_n_splits(X,y)
    for train_index, test_index in strat.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if metric == "auroc": #use decision_function in auroc, not predict
            y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        #print("SCORE:", score)
        if (score is not None):
            scores.append(score) 
    #print("scores =", scores)
    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with 
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    maxc = 0
    maxperf = 0
    for c in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = True, C = c, random_state = 445)
        perf = cv_performance(clf, X, y, k = 5, metric = metric)
        print(c, perf)
        if perf > maxperf:
            maxperf = perf
            maxc = c
    return maxc


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    for c in C_range:
        # clf = LinearSVC(penalty = penalty, loss = loss, dual = True, c = c, random_state = 445)
        clf = LinearSVC(penalty = penalty, loss = loss, C = c, dual = dual, random_state = 445)
        clf.fit(X, y)
        L0_norm = 0
        for theta in clf.coef_:
            for c in theta:
                if c != 0:
                    L0_norm += 1
        norm0.append(L0_norm)


    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM 
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    maxperf = 0
    for c, r in param_range:
        clf = SVC(kernel= "poly", degree=2, C=c, coef0=r, gamma="auto")
        #clf = SVC(kernel = 'poly', degree = 2, C = c,  coef0 = r, gamma = 'auto') 
        perf = cv_performance(clf, X, y, k = k, metric = metric)
        print(c, r, perf)
        if perf > maxperf:
            best_C_val = c
            best_r_val = r
            maxperf = perf 
    return best_C_val, best_r_val


def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )

    #print(extract_word("It's a test sentence. Does it look correct?"))

    # TODO: Questions 3, 4, 5

    #3.b
    print("number of unique words, d, = ", len(X_train[0]))

    #3.c 
    print('Avg number of non-zero features = ', np.sum(X_train)/len(X_train))
    #word appearing in greatest number of comments
    most_common_word = max(dictionary_binary, key = dictionary_binary.get)
    print("Most common word =", most_common_word)
    #4.1b
    print("4.1b ---------------------------------------------------------")
    metrics = ["accuracy", "precision", "sensitivity", "specificty", "f1-score", "auroc"]
    selected_C = 0
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for m in metrics:
        maxc = select_param_linear(X_train, Y_train, penalty = "l2", 
                        loss = "hinge", metric = m, C_range = C_range)
        clf = LinearSVC(penalty = "l2", loss = "hinge", dual = True, C = maxc, random_state = 445) 
        score = cv_performance(clf, X_train, Y_train, metric = m)
        print("C = ", maxc, "is optimal under", m, "metric, cv_performance =", score)
        if m == "auroc":
            selected_C = maxc

    #4.1c
    #train performance on X_test and Y_Test
    clf = LinearSVC(penalty = "l2", loss = "hinge", dual = True, C = selected_C)
    clf.fit(X_train, Y_train)
    Y_pred = clf.decision_function(X_test)
    auroc_score = performance(Y_test, Y_pred, metric = 'auroc')
    print("The C that maximizes AUROC is", selected_C)
    print("AUROC score: ", auroc_score)
    Y_pred = clf.predict(X_test)
    for m in metrics:
        if m != "auroc":
            score = performance(Y_test, Y_pred, metric = m)
            print("The", m, "score is", score)
    
    #4.1d 
    plot_weight(X_train, Y_train, penalty = "l2", loss = "hinge", dual = True, C_range = C_range)

    #4.1e 
    #bar coefficient vs each word, most pos and most neg, C = 0.1
    print("4.1e------------------------------------------------------------------------")
    clf = LinearSVC(C = 0.1)
    clf.fit(X_train, Y_train)
    arg = clf.coef_[0].argsort()
    min_ind5 = arg[:5]
    max_ind5 = arg[:-6:-1]
    minwords = []
    maxwords = []

    for ind in min_ind5:
        for word, index in dictionary_binary.items():
            if index == ind:
                minwords.append(word) 
    print("Most negative words")
    for i in range(5): #Return 5 most negative words
        print(clf.coef_[0][min_ind5[i]], minwords[i])


    plt.bar(minwords, clf.coef_[0][min_ind5])
    plt.xlabel("Most Negative Words")
    plt.ylabel("Corresponding Coefficient")
    plt.title("Bar Chart of most negative words.png")
    plt.savefig("bar Chart of most negative words.png")
    plt.close()
    
    
    for ind in max_ind5:
        for word, index in dictionary_binary.items():
            if index == ind: 
                maxwords.append(word)
    print("Most positive words")
    for i in range(5): #Return 5 most positive words
        print(clf.coef_[0][max_ind5[i]], maxwords[i])

    #plt.bar(maxwords, max_ind5, color = "blue")
    #plt.xlabel("Most Positive Words")
    #plt.ylabel("Corresponding Coefficient")
    #plt.show()
        
    plt.bar(maxwords, clf.coef_[0][max_ind5])
    #plt.xscale("Words")
    #plt.legend(["L0-norm"])
    plt.xlabel("Most Positive Words")
    plt.ylabel("Corresponding Coefficient")
    plt.title("Bar Chart of most positive words.png")
    plt.savefig("bar Chart of most positive words.png")
    plt.close()
    
    #4.2 
    #Use squared hinge and l1, dual = false
    
    #4.2a, reset maxc and maxperf
    maxc = 0 
    maxperf = 0 
    for c in C_range: 
        clf = LinearSVC(penalty = "l1", loss = "squared_hinge", C = c, dual = False)
        clf.fit(X_train, Y_train)
        y_pred = clf.decision_function(X_test)
        perf = performance(Y_test, Y_pred, "auroc")
        CV_auroc = cv_performance(clf, X_train, Y_train, metric = "auroc")
        if perf > maxperf: 
            maxc = c
            maxperf = perf
    print("4.2a--------------------------------------------------------------------------------")
    print("C = ", maxc, "is the optimal solution with an auroc score of",
             perf, "and a CV auroc score of", CV_auroc)

    #4.2b plot weight
    plot_weight(X_train, Y_train, penalty = "l1", loss = "squared_hinge", 
                C_range = C_range, dual = False)
    
    #4.3a
    #(i) Grid Search
    r_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    cr_range = []
    for c in C_range:
        for r in r_range:
            cr_range.append([c,r])
    [maxc, maxr] = select_param_quadratic(X_train, Y_train, param_range = cr_range)

    clf1 = SVC(kernel="poly", degree=2, C=c, coef0=r, gamma="auto")
    auroc_score_grid = cv_performance(clf1, X_train, Y_train, metric = "auroc")

    print("4.3a (i) Grid Search-------------------------------------------------")
    print("C = ", maxc, "r = ", maxr,
         "is part of the optimal solution with an auroc score of", auroc_score_grid)

    #(ii) Random Search
    import random
    cr_range = []
    for i in range(25):
        #range of values (-10e2, 10e3) --> 36 pairs
        lgc = random.uniform(-2, 3)
        lgr = random.uniform(-2, 3)
        cr_range.append([10**lgc, 10**lgr])
    [maxc, maxr] = select_param_quadratic(X_train, Y_train, param_range = cr_range)

    clf2 = SVC(kernel="poly", degree=2, C=c, coef0=r, gamma="auto")
    auroc_score_rand = cv_performance(clf2, X_train, Y_train, metric = "auroc")

    print("4.3a (ii) Random Search-------------------------------------------------")
    print("C = ", maxc, "r = ", maxr, "is the optimal solution with an auroc score of", 
                auroc_score_rand)


    #5.1c
    print("5.1c------------------------------------------------------------")
    clf = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight = {-1: 1, 1: 10})
    clf.fit(X_train, Y_train)
    Y_pred = clf.decision_function(X_test)
    perf = performance(Y_test, Y_pred, metric = "auroc")
    print("AUROC score: ", perf)
    Y_pred = clf.predict(X_test)
    for m in metrics: 
        if m != "auroc":
            perf = performance(Y_test, Y_pred, metric = m)
            print("Score: ", perf)

    #5.2a 
    print("5.2a-------------------------------------------------------")
    clf = LinearSVC(penalty = "l2", loss = "hinge", C = 0.01, class_weight = {-1: 1, 1 : 1})
    clf.fit(IMB_features, IMB_labels)
    Y_pred = clf.decision_function(IMB_test_features)
    perf = performance(IMB_test_labels, Y_pred, metric = "auroc")
    print("AUROC score: ", perf)
    Y_pred = clf.predict(IMB_test_features)
    #print("Y_pred:", np.shape(Y_pred))
    #print("IMB:", np.shape(IMB_test_labels))
    for m in metrics:
        if m != "auroc":
            perf = performance(IMB_test_labels, Y_pred, metric = m)
        print("Score: ", perf)
    

    #5.3a
    W_range = [-2, -1, 0, 1, 2]
    W_range = [10**w for w in W_range]
    maxWn = 0
    maxWp = 0
    maxperf = 0
    for Wn in W_range:
        for Wp in W_range:
            clf = SVC(C = 1, class_weight = {-1:Wn, 1:Wp})
            perf = cv_performance(clf, IMB_features, IMB_labels, metric = "auroc")
            if perf > maxperf:
                maxperf = perf
                maxWn = Wn
                maxWp = Wp
    print("5.3a----------------------------------------------------------------------")
    print("Wn =", Wn, "is optimal and Wp =", Wp, "is optimal and performance = ", maxperf)
    
    #based on values from 5.3a, the most optimal Wn value = 100,
    # the most optimal Wp value = 100 and performance = 0.9528 
    clf = SVC(C=1, class_weight = {-1:100, 1:100})
    perf = cv_performance(clf, IMB_features, IMB_labels, metric = "auroc")

    #5.3b
    metrics = ["accuracy", "precision", "sensitivity", "specificity", "f1-score", "auroc"]
    print("5.3b---------------------------------------------------------------")
    print("the auroc score:", maxperf) #maxperf comes from 5.3a
    clf = SVC(C = 1, class_weight = {-1:maxWn, 1:maxWp})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.predict(IMB_test_features)
    for m in metrics:
        if m != "auroc":
            perf = performance(IMB_test_labels, y_pred, metric = m)
            print("Metric", m, " =", perf)
    

    #5.4
    print("5.4---------------------------------------")
    clf = SVC(C=1, class_weight = {-1:100, 1:100})
    clf.fit(IMB_features, IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr, tpr, threshhold1 = metrics.roc_curve(IMB_test_labels, y_pred)
    perf = cv_performance(clf, IMB_features, IMB_labels, metric = "auroc")

    ROCclf = SVC(C = 0.01, class_weight = {-1:1, 1:1})
    ROCclf.fit(IMB_features, IMB_labels)

    y_pred2 = ROCclf.decision_function(IMB_test_features)
    fpr2, tpr2, threshold2 = metrics.roc_curve(IMB_test_labels, y_pred2)
    perf2 = performance(IMB_test_labels, y_pred2, metric = "auroc")

    plt.figure()
    plt.plot(fpr, tpr, color = "blue", label = "Custom Wn and Wp" % perf)
    plt.plot(fpr2, tpr2, color = "red", label = "Balanced Wn and Wp" % perf2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc = "lower right")
    plt.xlabel("FP (false positive) rate")
    plt.ylabel("TP (true positive) rate")
    plt.title("ROC Curve.png")
    plt.savefig("ROC Curve.png")
    plt.close()


    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()
    
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    #decision_function_shape = "ovo"
    clf = SVC(C = 0.1, kernel = "rbf", gamma = "auto", class_weight = "balanced", 
                decision_function_shape = 'ovo')
    clf.fit(multiclass_features, multiclass_labels)
    y_pred = clf.predict(heldout_features)
    generate_challenge_labels(y_pred, "cnrusimh")
    

if __name__ == "__main__":
    main()
