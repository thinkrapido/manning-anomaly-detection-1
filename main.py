import sys
import os

from math import inf
import numpy as np
import matplotlib
matplotlib.use("GTK4Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# def roc(true_labels, predictions):
#     """
#     -------------------------------------------------------------------------------------------------------------------------------
#     Args:
#         true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

#         predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

#     Returns:
#         false positive rate and corresponding true positive rate arrays.
#     -------------------------------------------------------------------------------------------------------------------------------
#     """

#     POSITIVE = 1

#     # 1. count total number of positive salmples

#     # 2. count total number of negative sample

#     # 3. find indices that sort the predictions array

#     # 4. sort the same way the predictions and the true_labels

#     f_prev = -inf

#     # 5. define a way to store the false positive and true positive rates


#     # keep counts 
#     fp = 0.0

#     tp = 0.0

#     while idx < num_examples:

#         # 6. condition to decide if the false positive and true positive rates should be updated
#         if <CONDITION>:

#             # 7. update important quantities

#         if # 8. condition to update fp and tp counts:
#             tp = <CODE>
#         else:
#             fp = <CODE>

#         idx += 1

#     # 9. don't miss a case here, bewew of +/- 1 count errors  

#     return # false positive rate and true positive rate


# def roc_conceptual(true_labels, predictions, number_of_thresholds=1000):

#     # 1. min prediction value

#     # 2. max prediction values

#     # 3. threshold points to be considered, 
#     # the higher the value of number_threshold_points, the more threshold points should be sampled 
#     thresholds = <CODE>

#     # 4. somewhere to store the important results


#     # definition of the positive class
#     POSITIVE = 1

#     # 5. Find the positive and negative cases
#     pos = <CODE>

#     neg = <CODE>

#     # 6. Count the number of positive and negative cases
#     num_positive_examples = <CODE>

#     num_negative_examples = <CODE>

#     for i, t in enumerate(thresholds):

#         # 7. for every threshold get the false positive and true positive rates, pos and neg can be helpful
        
#     return #8. false positive rate and true positive rate



# def auc_roc(true_labels, predictions):
#     """
#     -------------------------------------------------------------------------------------------------------------------------------
#     Args:
#         true_labels: numpy array of true labels, positive class is signaled by a value of 1, any other value is assumed as negative class

#         predictions: model class predictions. Higher values denote more likelihood of sample belonging to the positive class

#     Returns:
#         AUC Area under de ROC curve.
#     -------------------------------------------------------------------------------------------------------------------------------
#     """

#     # 1. get the false and true positive rates
#     fpr, tpr = roc(true_labels, predictions)

#     return # 2. use a numpy function to perfom the numerical integration


if __name__ == '__main__':

    # Set up a random classification problem

    X, y = make_classification(random_state=87817)  # 8787917 #1231
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(random_state=0).fit(X_train, y_train)
    y_pred = clf.decision_function(X_test)

    # plot and display the sklearn ROC and AUC results

    RocCurveDisplay.from_predictions(y_test, y_pred,name="sklearn")
    
    # calls your roc and auc implementations

    # fpr, tpr = roc(y_test, y_pred)
    # roc_auc_aduh = auc_roc(y_test, y_pred)
    # impl = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_aduh, estimator_name="Lerner")
    # impl.plot()

    plt.ion()
    plt.show()
    plt.waitforbuttonpress()
