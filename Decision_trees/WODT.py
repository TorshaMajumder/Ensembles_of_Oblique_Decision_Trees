#
#
# Implementation of Weighted Oblique Decision Trees by [Bin-Bin Yang et al.]
# ````Some part of the code has been shared by the author````
#
#
#.......Importing all the packages.............
#
import random
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
#
# Global variables
#
epsilonepsilon = 1e-220
epsilon = 1e-50
#
# Class for splitting the data set based on the equation (wx <= b)
#
class SplitQuestion(object):

	def __init__(self, attrIDs=[0], paras=[0], threshold=0):
		super(SplitQuestion, self).__init__()
		self.attrIDs = attrIDs
		self.paras = paras
		self.threshold = threshold

	# we only consider continuous attributes for simplicity
	def test_forOneInstance(self, x):
		return np.dot(x[self.attrIDs], self.paras) <= self.threshold

	def test(self, X):
		return np.dot(X[:, self.attrIDs], self.paras) <= self.threshold

class node(object):													             # A Node definition

	def __init__(self, depth, split, sample_ids, X, Y, class_num):
		super(node, self).__init__()
		self.sample_ids = sample_ids
		self.split = split
		self.depth = depth
		self.X = X
		self.Y = Y
		self.class_num = class_num
		self.is_leaf = False
		# after grow_stump, set the node as an internal node

	def find_best_split(self, max_features='sqrt'):
		feature_num = self.X.shape[1]
		subset_feature_num = feature_num
		if max_features == 'sqrt':
			subset_feature_num = int(np.sqrt(feature_num))
		if max_features == 'all':
			subset_feature_num = feature_num
		if max_features == 'log':
			subset_feature_num = int(np.log2(feature_num))
		if isinstance(max_features, int):
			subset_feature_num = max_features
		if isinstance(max_features, float):
			subset_feature_num = int(feature_num * max_features)

		feature_ids = range(feature_num)
		subset_feature_ids = random.sample(feature_ids, subset_feature_num)		# get random subset of features
		self.split.attrIDs = subset_feature_ids									# feature 0 is threshold
		subset_feature_ids = np.array(subset_feature_ids)

		X = self.X
		subFeatures_X = X[self.sample_ids[:, None], subset_feature_ids[None, :]]
		Y = self.Y[self.sample_ids]
		class_num = self.class_num

		def func(a):															# define func and func_gradient for optimization
			paras = a[1:]
			threshold = a[0]
			p = sigmoid(np.dot(subFeatures_X, paras) - threshold)
			w_R = p
			w_L = 1 - w_R
			w_R_sum = w_R.sum()
			w_L_sum = w_L.sum()
			w_R_eachClass = np.array([sum(w_R[Y == k]) for k in range(class_num)])
			w_L_eachClass = np.array([sum(w_L[Y == k]) for k in range(class_num)])
			fun = w_L_sum * np.log2(w_L_sum + epsilonepsilon) \
					+ w_R_sum * np.log2(w_R_sum + epsilonepsilon) \
					- np.sum(w_R_eachClass * np.log2(w_R_eachClass + epsilonepsilon)) \
					- np.sum(w_L_eachClass * np.log2(w_L_eachClass + epsilonepsilon))

			return fun

		def func_gradient(a):
			paras = a[1:]
			threshold = a[0]

			p = sigmoid(np.dot(subFeatures_X, paras) - threshold)
			w_R = p
			w_L = 1 - w_R
			w_R_eachClass = np.array([sum(w_R[Y == k]) for k in range(class_num)])
			w_L_eachClass = np.array([sum(w_L[Y == k]) for k in range(class_num)])
			la = np.log2(w_L_eachClass[Y] * w_R.sum() + epsilonepsilon) - np.log2(w_R_eachClass[Y] * w_L.sum() + epsilonepsilon)
			beta = la * p * (1 - p)

			jac = np.zeros(a.shape)
			jac[0] = - np.sum(beta)
			jac[1:] = np.dot(subFeatures_X.T, beta)

			return jac

		################################################
		initial_a = np.random.rand(subset_feature_num + 1) - 0.5
		result = minimize(func, initial_a, method='L-BFGS-B', jac=func_gradient,\
			options={'maxiter':10, 'disp': False})

		##########################################
		self.split.paras = result.x[1:]
		self.split.threshold = result.x[0]

		return 1


	def grow_stump(self):
		L_bool = self.split.test(self.X[self.sample_ids])
		L_sample_ids = self.sample_ids[L_bool]
		R_sample_ids = self.sample_ids[~L_bool]
		LChild = node(self.depth + 1, SplitQuestion(), L_sample_ids, self.X, self.Y, self.class_num)
		RChild = node(self.depth + 1, SplitQuestion(), R_sample_ids, self.X, self.Y, self.class_num)

		if len(L_sample_ids) == 0:
			LChild.is_leaf = True
			LChild.class_distribution = compute_class_distribution(self.Y[self.sample_ids], self.class_num)
		if len(R_sample_ids) == 0:
			RChild.is_leaf = True
			RChild.class_distribution = compute_class_distribution(self.Y[self.sample_ids], self.class_num)

		self.LChild = LChild
		self.RChild = RChild



class BaseObliqueTree(BaseEstimator):
    def __init__(self, max_depth, min_samples_split, max_features):

        # Get the options for tree learning
        self.max_depth = max_depth                     # maximum depth of the tree
        self.min_samples_split = min_samples_split     # number of features to consider when looking for the best split
        self.max_features = max_features               # remove features that occur in previous nodes

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.classNum = self.Y.max() + 1
        self.sampleNum = self.X.shape[0]
        self.root_node = node(1, SplitQuestion(), np.arange(self.sampleNum, dtype=np.uint32), self.X, self.Y, self.classNum)
        self.leaf_num = 1
        self.tree_depth = self.build_subtree(self.root_node, is_classifier(self))


    def build_subtree(self, node, is_classification):
        ## the tree grows up until each leaf node are class-pure, the default max tree depth is 50.
        if node.is_leaf:
            return node.depth

        # stopping conditions
        is_leaf = node.depth >= self.max_depth or \
                  len(node.sample_ids) < self.min_samples_split or \
                  is_all_equal(self.Y[node.sample_ids])

        if is_leaf or node.find_best_split(self.max_features) < 0:
            node.is_leaf = True
            if is_classification:
                node.class_distribution = compute_class_distribution(self.Y[node.sample_ids], self.classNum)
                return node.depth


        node.grow_stump()
        node.is_leaf = False
        self.leaf_num += 1
        L_subtree_depth = self.build_subtree(node.LChild, is_classification)
        R_subtree_depth = self.build_subtree(node.RChild, is_classification)
        return max(L_subtree_depth, R_subtree_depth)

    def predict_forOneInstance(self, x):
        present_node = self.root_node
        while not (present_node.is_leaf):
            if present_node.split.test_forOneInstance(x):
                present_node = present_node.LChild
            else:
                present_node = present_node.RChild
        return np.argmax(present_node.class_distribution)

    def predict(self, X):
        m = X.shape[0]
        Y_predicted = np.zeros((m,), dtype=int)
        for i in range(m):
            x = X[i]
            Y_predicted[i] = self.predict_forOneInstance(x)
        return Y_predicted

####################
'''function'''
def sigmoid(z):
    # because that -z is too big will arise runtimeWarning in np.exp()
    if isinstance(z, float) and (z < -500):
        z = -500
    elif not (isinstance(z, float)):
        z[z < -500] = (-500) * np.ones(sum(z < -500))

    return 1 / (np.exp(- z) + 1)


def is_all_equal(x):
    x_min, x_max = x.min(), x.max()
    return (x_min == x_max)


def compute_class_distribution(Y, class_num):
    sample_num = len(Y)
    ratio_each_class = [sum(Y == k) / sample_num for k in range(class_num)]
    return np.array(ratio_each_class)


#
# Definition of classes provided: WeightedObliqueDecisionTreeClassifier
#
class WeightedObliqueDecisionTreeClassifier(ClassifierMixin, BaseObliqueTree):
    def __init__(self, max_depth=50, min_samples_split=2, max_features='all'):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)


