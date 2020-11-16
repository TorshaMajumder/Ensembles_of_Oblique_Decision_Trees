#
#
# Implementation of Non-Linear Decision Trees by [Ittner et al.]
#
#
#
#......Importing all the packages.................
#
from Oblique_Classifier_1 import *
import numpy as np
from itertools import combinations


class BaseObliqueTree(BaseEstimator):


    def __init__(self, criterion, max_depth, min_samples_split, min_features_split):

        self.criterion = criterion                     # splitting criterion
        self.max_depth = max_depth                     # maximum depth of the tree
        self.min_samples_split = min_samples_split     # number of samples to consider when looking for the best split
        self.min_features_split = min_features_split   # number of features to consider when looking for the best split
        self.clf = ObliqueClassifier1(criterion=self.criterion, max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_features_split=self.min_features_split) # Oblique Classifier 1 by [Murthy et al.]

    def fit(self, X, y):
        X = X
        y = y
        X = NewFeatureSpace(X)
        self.clf.fit(X, y)

    def predict(self, X):
        X = NewFeatureSpace(X)
        y = self.clf.predict(X)
        y = np.array(y, dtype=int)
        return y

#
# Definition of classes provided: NDTClassifier
#
class NDTClassifier(ClassifierMixin, BaseObliqueTree):
    def __init__(self, criterion="gini", max_depth=3, min_samples_split=2, min_features_split=1):
        super().__init__(criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_features_split=min_features_split)

#
# Generating new features for the oblique decision tree
#
def NewFeatureSpace(X):

    comb = combinations(np.arange(len(X[0, :])), 2)
    new_X = X
    for i in list(comb):
        new_feature = X[:,i[0]]*X[:,i[1]]
        new_feature = np.array(new_feature)
        new_X =np.column_stack((new_X, new_feature))

    for i in range(len(X[0, :])):
        new_feature = X[:, i] * X[:, i]
        new_feature = np.array(new_feature)
        new_X = np.column_stack((new_X, new_feature))

    return new_X

