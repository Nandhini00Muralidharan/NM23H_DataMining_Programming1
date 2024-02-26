"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


import numpy as np


def scale_data(data):
    return data/255.0

def perform_cross_validation(X, y, n_splits, seed):
    clf = DecisionTreeClassifier(random_state=seed)
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    cv_results = cross_validate(clf, X, y, cv=cv, scoring='accuracy')
    mean_accuracy = np.mean(cv_results['test_score'])
    std_accuracy = np.std(cv_results['test_score'])
    return mean_accuracy, std_accuracy


def filter_and_modify_7_9s(X: np.ndarray, y: np.ndarray):
    """
    Filter the dataset to include only the digits 7 and 9, then modify the labels.
    Parameters:
        X: Data matrix
        y: Labels
    Returns:
        Filtered and modified data matrix and labels
    """
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx, :]
    y_binary = y[seven_nine_idx]

    y_binary[y_binary == 7] = 0

    nine_indices = np.where(y_binary == 9)[0]

    convert_nines = np.random.choice(nine_indices, size=int(0.1 * len(nine_indices)), replace=False)
    y_binary[convert_nines] = 1

    delete_nines = np.setdiff1d(nine_indices, convert_nines)
    X_binary = np.delete(X_binary, delete_nines, axis=0)
    y_binary = np.delete(y_binary, delete_nines)

    return X_binary, y_binary