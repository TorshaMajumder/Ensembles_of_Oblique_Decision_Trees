#
#
# Implementation of Continuous Optimization of Oblique Splits (CO2) by [Norouzi et al.]
#
#
#
#.......Importing all the packages...........
#
import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
#
#
#
class CO2Node:                                                      # defining the CO2 Node

    def __init__(self, depth, labels, **kwargs):
        super(CO2Node, self).__init__()
        self.depth = depth
        self.labels = labels
        self.is_leaf = kwargs.get('is_leaf', False)
        self._weights = kwargs.get('weights', None)
        self._bias = kwargs.get('bias', None)
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)
        self._probl = kwargs.get('pl', None)
        self._probr = kwargs.get('pr',None)


        if not self.is_leaf:
            assert self._left_child
            assert self._right_child


    def get_child(self, datum):
        if self.is_leaf:
            raise Warning("Leaf node does not have children.")
        X = deepcopy(datum)
        if (np.dot(X, self._weights) + self._bias) < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if not hasattr(self, '_label'):
            classes, counts = np.unique(self.labels, return_counts=True)
            self._label = classes[np.argmax(counts)]
        return self._label



    @property
    def left_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._right_child



class ContinuouslyOptimizedObliqueTree(BaseEstimator):

    def __init__(self, impurity, segmentor, max_depth, min_samples_split=2,nu=1.0, tau=10, tol=1e-3, eta=0.1):
        self.impurity = impurity
        self.segmentor = segmentor
        self.nu = nu
        self.tau = tau
        self._max_depth = max_depth
        self._min_samples = min_samples_split
        self.tol = tol
        self.eta = eta
        self._root = None
        self._nodes = []


    def _terminate(self, X, y, cur_depth):                                  # conditions for termination
        if self._max_depth != None and cur_depth == self._max_depth:        # maximum depth reached
            return True
        elif y.size < self._min_samples:                                    # minimum number of samples reached
            return True
        elif np.unique(y).size == 1:                                        # node is homogeneous
            return True
        else:
            return False

    def _generate_leaf_node(self, cur_depth, y):
        node = CO2Node(cur_depth, y, is_leaf=True)
        self._nodes.append(node)
        return node

    def likelihood_softmax(self, theta, y):
        return -theta[y] + np.log(np.sum(np.exp(theta)))

    def objective(self, X, y, w, thetaL, thetaR):

        labels_y = np.unique(y)
        index_y_samples = np.zeros(len(y), dtype=int)
        for i in labels_y:
            index = np.where(y == i)
            index_y_samples[index] = np.where(labels_y == i)[0]

        loglikL = self.likelihood_softmax(thetaL, index_y_samples)
        loglikR = self.likelihood_softmax(thetaR, index_y_samples)
        boundL = -np.dot(X, w) + loglikL
        boundR = np.dot(X, w) + loglikR
        loss = np.vstack((boundL, boundR))
        objective = (np.max(loss, axis=0) - np.abs(np.dot(X, w))).sum()
        return objective

    def _generate_node(self, X, y, cur_depth):
        if self._terminate(X, y, cur_depth):
            return self._generate_leaf_node(cur_depth, y)
        else:
            n_samples, n_features = X.shape
            class_labels = np.unique(y)
            n_classes = class_labels.shape[0]
            #
            # Initialize the parameters of the left node (thetaL) and right node (thetaR)
            #
            probL, probR = np.full((n_classes,), 1 / n_classes), np.full((n_classes,), 1 / n_classes)
            thetaL, thetaR = np.full((n_classes,), 1 / n_classes), np.full((n_classes,), 1 / n_classes)
            #
            # Initialize the parameters of the split (w), this is augmented by 1 for the bias
            #
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y)
            w = np.hstack([np.eye(1, n_features-1, tree.tree_.feature[0])[0], -tree.tree_.threshold[0]])
            converged = False
            #
            # Check for convergence
            #
            while not converged:
                w_old = np.copy(w)
                old_objective = self.objective(X, y, w_old, thetaL, thetaR)
                for t in range(1, (self.tau + 1)):
                    sample = np.random.choice(n_samples, size=1)                           # considering all the samples
                    labels_y = np.unique(y[sample])
                    index_y_samples = np.zeros(len(y[sample]), dtype=int)
                    y_sample = y[sample]
                    x_sample = X[sample]
                    for i in labels_y:
                        index = np.where(y_sample == i)
                        index_y_samples[index] = np.where(class_labels == i)[0]
                    for i in range(len(sample)):
                        s = np.sign(np.dot(w_old, X[i, :]))
                        loglikL = self.likelihood_softmax(thetaL, index_y_samples[i])
                        loglikR = self.likelihood_softmax(thetaR, index_y_samples[i])
                        boundL = -np.dot(w, X[i, :]) + loglikL
                        boundR = np.dot(w, X[i, :]) + loglikR
                        c = np.squeeze(np.eye(n_classes)[index_y_samples[i].reshape(-1)])  # One hot vector of the label
                        if boundL >= boundR:
                            w += (self.eta / np.sqrt(t)) * (1 + s) * x_sample[i, :]
                            probL = np.exp(thetaL) / np.sum(np.exp(thetaL))
                            thetaL -= (self.eta / np.sqrt(t)) * (probL - c)

                        else:
                            w -= (self.eta / np.sqrt(t)) * (1 - s) * x_sample[i, :]
                            probR = np.exp(thetaR) / np.sum(np.exp(thetaR))
                            thetaR -= (self.eta / np.sqrt(t)) * (probR - c)


                        if np.linalg.norm(w) ** 2 > self.nu:
                            w *= np.sqrt(self.nu) / np.linalg.norm(w)



                # Check for convergence
                new_objective = self.objective(X, y, w, thetaL, thetaR)
                if (np.abs(old_objective - new_objective) < 1e-2):
                    # if np.linalg.norm(w - w_old) < 1e-4:
                    converged = True
                    probL = np.exp(thetaL) / np.sum(np.exp(thetaL))
                    probR = np.exp(thetaR) / np.sum(np.exp(thetaR))
            weights = w[:-1]
            bias = w[-1]
            pl= probL
            pr= probR
            mask = ((np.dot(X,w)) >= 0)
            X_left, y_left = X[np.logical_not(mask)], y[np.logical_not(mask)]
            X_right, y_right = X[mask], y[mask]
            if (len(y_right) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            elif (len(y_left) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y)
            else:
                node = CO2Node(cur_depth, y,
                               weights=weights, bias=bias, pl = pl, pr= pr,
                               left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                               right_child=self._generate_node(X_right, y_right, cur_depth + 1),
                               is_leaf=False)
                self._nodes.append(node)

                return node

    def fit(self, X, y):

        n_samples, n_features = X.shape
        X = np.hstack([X, np.ones((n_samples, 1))])                         # Augment the examples with a vector of ones

        self._root = self._generate_node(X, y, 0)

    def get_params(self, deep=True):
        return {'max_depth': self._max_depth,
                'min_samples_split': self._min_samples,
                'impurity': self.impurity, 'segmentor': self.segmentor, 'nu': self.nu, 'tau':self.tau ,'tol': self.tol,
                'eta': self.eta}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):

        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label

        if not self._root:
            raise Warning("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size,), dtype=int)
        for i in range(size):
            predictions[i] = predict_single(X[i, :])
        return predictions
#
# Definition of classes provided: CO2Classifier
#
class CO2Classifier(ClassifierMixin, ContinuouslyOptimizedObliqueTree):
    def __init__(self, impurity, segmentor, max_depth=50, min_samples_split=2, nu=4.0, tau=10, tol=1e-3, eta=0.01):
        super().__init__(impurity=impurity, segmentor=segmentor, max_depth=max_depth, min_samples_split=min_samples_split,nu=nu, tau=tau, tol=tol, eta=eta)