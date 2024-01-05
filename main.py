import sys
import os

from math import inf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def roc(true_labels, predictions):

    actual = np.array(true_labels)
    y_pred = np.array(predictions)

    idxs = np.argsort(predictions)
    
    actual = actual[idxs]
    y_pred = y_pred[idxs]

    tpr, fpr = [], []

    for threshold in np.unique(y_pred):
        
        prediction = y_pred >= threshold

        tp = np.sum((actual == 1) & (prediction == 1))
        tn = np.sum((actual == 0) & (prediction == 0))
        fp = np.sum((actual == 0) & (prediction == 1))
        fn = np.sum((actual == 1) & (prediction == 0))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    return fpr, tpr



def auc_roc(true_labels, predictions):
    """
    -------------------------------------------------------------------------------------------------------------------------------
    Args:
        true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

        predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

    Returns:
        AUC Area under de ROC curve.
    -------------------------------------------------------------------------------------------------------------------------------
    """

    # 1. get the false and true positive rates
    fpr, tpr = roc(true_labels, predictions)

    return 1 + np.trapz(fpr, tpr)


if __name__ == '__main__':

    # Set up a random classification problem

    X, y = make_classification(random_state=87817)  # 8787917 #1231
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(random_state=0).fit(X_train, y_train)
    y_pred = clf.decision_function(X_test)

    # plot and display the sklearn ROC and AUC results

    RocCurveDisplay.from_predictions(y_test, y_pred, name="sklearn")
    
    # calls your roc and auc implementations

    fpr, tpr = roc(y_test, y_pred)
    roc_auc_aduh = auc_roc(y_test, y_pred)
    impl = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_aduh, estimator_name="Lerner")
    impl.plot()

    plt.ion()
    plt.show()
    plt.waitforbuttonpress()
