# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, cross_validate, KFold, train_test_split
from sklearn.metrics import confusion_matrix

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        #LOAD DATASET
        X, y = self.load_data()
        
        #SPLIT - TRAINING AND TESTING SET
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        #NUMBER OF UNIQUE CLASSES, COUNT OF EACH CLASS
        classes_train, count_train = unique(ytrain, return_counts=True)
        classes_test, count_test = unique(ytest, return_counts=True)
        
        answer = {
            'nb_classes_train': len(classes_train),
            'nb_classes_test': len(classes_test),
            'class_count_train': dict(zip(classes_train, count_train)),
            'class_count_test': dict(zip(classes_test, count_test)),
            'length_Xtrain': Xtrain.shape[0],
            'length_Xtest': Xtest.shape[0],
            'length_ytrain': ytrain.shape[0],
            'length_ytest': ytest.shape[0],
            'max_Xtrain': Xtrain.max(),
            'max_Xtest': Xtest.max()
        }

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        
        for i in range(0, len(ntrain_list)):
            ntrain = ntrain_list[i]
            ntest = ntest_list[i]
    
            Xtrain = X[0:ntrain,:]
            ytrain = y[0:ntrain]
            Xtest = Xtest[0:ntest]
            ytest = ytest[0:ntest]
            
            X = Xtrain
            y = ytrain

            #PART C
            clf_C = DecisionTreeClassifier(random_state=self.seed)
            cv_C = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_results_C = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_C, cv=cv_C)
                    
            partC = {}
            scores_C= {}

            scores_C['mean_fit_time'] = cv_results_C['fit_time'].mean()
            scores_C['std_fit_time'] = cv_results_C['fit_time'].std()
            scores_C['mean_accuracy'] = cv_results_C['test_score'].mean()
            scores_C['std_accuracy'] = cv_results_C['test_score'].std()

            partC['clf'] = clf_C
            partC['cv'] = cv_C
            partC['scores'] = scores_C
                    
            #PART F
            partF ={}
        
            clf_F = LogisticRegression(max_iter=300, random_state=self.seed)
            cv_F = ShuffleSplit(n_splits=5, random_state=self.seed)
            scores_F = cross_validate(clf_F, X, y, cv=cv_F, return_train_score=True)
            
            clf_F.fit(X, y)
            
            scores_train_F = clf_F.score(X, y)
            scores_test_F = clf_F.score(Xtest, ytest) 
            
            train_prediction =clf_F.predict(X)
            test_prediction = clf_F.predict(Xtest)
            conf_mat_train = confusion_matrix(y, train_prediction)
            conf_mat_test = confusion_matrix(ytest, test_prediction)
            
            mean_cv_accuracy_F = scores_F["test_score"].mean()
            partF = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": clf_F,
                "cv": cv_F,
                "conf_mat_train": conf_mat_train,
                "conf_mat_test": conf_mat_test
            }
                    
            #PART D
            clf_D = DecisionTreeClassifier(random_state=self.seed)
            cv_D = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_results_D = u.train_simple_classifier_with_cv(Xtrain=Xtrain, ytrain=ytrain, clf=clf_D, cv=cv_D)
                    
            partD = {}
            scores_D= {}

            scores_D['mean_fit_time'] = cv_results_D['fit_time'].mean()
            scores_D['std_fit_time'] = cv_results_D['fit_time'].std()
            scores_D['mean_accuracy'] = cv_results_D['test_score'].mean()
            scores_D['std_accuracy'] = cv_results_D['test_score'].std()

            partD['clf'] = clf_D
            partD['cv'] = cv_D
            partD['scores'] = scores_D
            
            partD['explain_kfold_vs_shuffle_split'] = "K-Fold Cross-Validation divides the dataset into k sequential folds, utilizing each fold once as a test set while the remaining k-1 folds serve as the training set. This method ensures that every data point gets an opportunity to be in both the training and test sets, making it advantageous for smaller datasets where maximizing training data is crucial. On the other hand, Shuffle-Split Cross-Validation creates multiple independent train/test splits by shuffling the samples and dividing them into training and test sets. This technique offers more flexibility in determining the size of the test set and the number of iterations. It is particularly useful for larger datasets or when a more randomized selection of samples is desired. Advantages of Shuffle-Split include greater control over test set size and iteration numbers, making it efficient for large datasets. However, it may provide less systematic coverage of data compared to k-fold and could lead to higher variance in test performance across iterations due to its random nature. In practice, Shuffle-Split often proves faster and may yield higher accuracy, but it's essential to consider the trade-offs between systematic coverage and randomness in data selection."

            unique_train, counts_train = np.unique(ytrain, return_counts=True)
            class_count_train = dict(zip(unique_train, counts_train))
                    
            unique_test, counts_test = np.unique(ytest, return_counts=True)
            class_count_test = dict(zip(unique_test, counts_test))

            answer[ntrain_list[i]] = {
                "partC": partC,
                "partD": partD,
                "partF": partF,
                "ntrain": ntrain,
                "ntest": ntest,
                "class_count_train": class_count_train,
                "class_count_test": class_count_test
            }
                    
        return answer