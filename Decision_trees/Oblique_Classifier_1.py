#
#
# Implementation of Oblique Classifier 1 (OC1) by [Murthy et al.]
#
#
#
#......Importing all the packages............................
#
from warnings import warn
import numpy as np
from scipy.stats import mode
from OC1_tree_structure import Tree, Node, LeafNode
import split_criteria
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
#
#
#
epsilon = 1e-6
#
# Definition of classes provided: ObliqueClassifier1
#
class BaseObliqueTree(BaseEstimator):


    def __init__(self, criterion, max_depth, min_samples_split, min_features_split):

        # Get the options for tree learning
        self.criterion = criterion                      # splitting criterion --- default is 'gini-index'
        self.max_depth = max_depth                      # maximum depth of the tree
        self.min_samples_split = min_samples_split      # minimum number of samples needed for a split
        self.min_features_split = min_features_split    # minimum number of features needed for a split
        self.tree_ = None                               # Internal tree --- initially set as 'None'


    # Find the minimum number of samples per split
    def get_min_samples_split(self, n_samples):
        if isinstance(self.min_samples_split, int):
            if self.min_samples_split < 2:
                self.min_samples_split = 2
                warn('min_samples_split specified incorrectly; setting to default value of 2.')
            min_samples = self.min_samples_split
        elif isinstance(self.min_samples_split, float):
            if not 0. < self.min_samples_split <= 1.:
                self.min_samples_split = 2 / n_samples
                warn('min_samples_split not between [0, 1]; setting to default value of 2.')
                min_samples = 2
            else:
                min_samples = int(np.ceil(self.min_samples_split * n_samples))
        else:
            warn('Invalid value for min_samples_split; setting to defalut value of 2')
            min_samples = 2

        return min_samples

    # Find the minimum number of features per split
    def get_min_features_split(self, n_features):
        if isinstance(self.min_features_split, int):
            if self.min_features_split < 1:
                self.min_features_split = 1
                warn('min_features_split specified incorrectly; setting to default value of 1.')
            min_features = self.min_features_split
        elif isinstance(self.min_features_split, float):
            if not 0. < self.min_features_split <= 1.:
                self.min_features_split = 1 / n_features
                warn('min_features_split not between [0, 1]; setting to default value of 1.')
                min_features = 1
            else:
                min_features = int(np.ceil(self.min_features_split * n_features))
        else:
            warn('Invalid value for min_features_split; setting to default value of 1')
            min_features = 1

        return min_features

    def fit(self, X, y):
        X = X
        y = y
        # Ensure that X is a 2d array of shape (n_samples, n_features)
        if X.ndim == 1:                 # single feature from all examples
            if len(y) > 1:
                X = X.reshape(-1, 1)
            elif len(y) == 1:           # single training example
                X = X.resphape(1, -1)
            else:
                ValueError('Invalid X and y')


        if self.criterion == 'gini':
            self.criterion = split_criteria.gini
        elif self.criterion == 'twoing':
            self.criterion = split_criteria.twoing
        else:
            ValueError('Unrecognized split criterion specified. Allowed split criteria are:\n'
                       '[classification] "gini": Gini impurity, "twoing": Twoing rule')



        n_samples, n_features = X.shape
        min_samples_split = self.get_min_samples_split(n_samples)
        min_features_split = self.get_min_features_split(n_features)
        #
        # Build a tree and get its root node
        #
        self.root_node, self.learned_depth = build_oblique_tree_oc1(X, y, is_classifier(self), self.criterion,
                                                          self.max_depth, min_samples_split, min_features_split)
        #
        # Create a tree object
        #
        self.tree_ = Tree(n_features=n_features, is_classifier=is_classifier(self))
        self.tree_.set_root_node(self.root_node)
        self.tree_.set_depth(self.learned_depth)

    def predict(self, X):
        y = self.tree_.root_node.predict(X)
        y = np.array(y, dtype=int)
        return y

    def get_params(self, deep = False):
        return {'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'criterion': self.criterion, 'min_features_split': self.min_features_split}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


#
# Definition of classes provided: ObliqueClassifier1
#
class ObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
    def __init__(self, criterion="gini", max_depth=3, min_samples_split=2, min_features_split=1):
        super().__init__(criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_features_split=min_features_split)





# Implements Murthy et al (1994)'s algorithm to learn an oblique decision tree via random perturbations
def build_oblique_tree_oc1(X, y, is_classification, criterion,
                           max_depth, min_samples_split, min_features_split,
                           current_depth=0, current_features=None, current_samples=None, debug=False):

    n_samples, n_features = X.shape

    # Initialize
    if current_depth == 0:
        current_features = np.arange(n_features)
        current_samples = np.arange(n_samples)

    # Score the current node
    if is_classification:
        majority, count = mode(y)  # NOTE that if there is more than one mode, scipy.mode() returns the smallest one
        label = majority[0]
        conf = count[0] / len(y)  # Classification confidence: fraction of examples with majority label
    else:
        std = np.std(y)
        label = np.mean(y)
        conf = np.sum((-std <= y) & (y <= std)) / n_samples  # Regression confidence: fraction of examples within 1 std

    # Check termination criteria, and return a leaf node if terminating
    if (current_depth == max_depth or            # max depth reached
            n_samples <= min_samples_split or    # not enough samples to split on
            n_features <= min_features_split or  # not enough features to split on
            conf >= 0.95):                       # node is very homogeneous

        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    # Otherwise, learn a decision node
    feature_splits = get_best_splits(X, y, criterion=criterion)         # Get the best split for each feature
    f = np.argmin(feature_splits[:, 1])                                 # Find the best feature to split on
    best_split_score = feature_splits[f, 1]                             # Save the score corresponding to the best split
    w, b = np.eye(1, n_features, f).squeeze(), -feature_splits[f, 0]    # Construct a (w, b) from the best split
                                                                        # X[f] <= s becomes 1. X[f] + 0. X[rest] - s <= 0


    stagnant = 0                                                        # Used to track stagnation probability (see below)
    for k in range(5):                                                  # Randomly attempt to perturb a feature
        m = np.random.randint(0, n_features)                            # Select a random feature (weight w[m]) to update
        idx = np.where(X[:,m] == 0)[0]
        if len(idx) != 0:
            X[idx,m] = epsilon

        wNew = np.array(w)                                              # Initialize wNew to w
        margin = (np.dot(X, wNew) + b)                                  # Compute the signed margin of all training examples
        u = (X[:, m]*w[m] - margin) / X[:, m]                           # Compute the residual of all training examples

        possible_wm = np.convolve(np.sort(u), [0.5, 0.5])[1:-1]         # Generate a list of possible values for new w[m]
        scores = np.empty_like(possible_wm)
        best_wm, best_wm_score = 0, np.inf                              # Find the best value for w[m]
        i = 0
        for wm in possible_wm:
            wNew[m] = wm                                                # Try (w = [w0, ..., wm, ..., wd], b) as a split
            margin = (np.dot(X, wNew) + b)                              # Signed margin of examples using this split
            left, right = y[margin <= 0], y[margin > 0]                 # Partition of examples using this split
            wm_score = criterion(left, right)                           # Score of this split
            scores[i] = wm_score
            i += 1

            if wm_score < best_wm_score:                                # Save the best split
                best_wm_score = wm_score
                best_wm = wm

        # Once the best w[m] among possible u has been identified, check if its actually any good
        if best_wm_score < best_split_score:
            # If we've identified a split with a better score, update it
            best_split_score = best_wm_score
            w[m] = best_wm
            stagnant = 0
        elif np.abs(best_wm_score - best_split_score) < 1e-3:
            # If we've identified a split with a similar score, update it with probability P(update) = exp(-stagnant)
            # Stagnation prob. is the probability that the perturbation does not improve the score. To prevent the
            # impurity from remaining stagnant for a long time, the stag. prob decreases exponentially with the number
            # of stagnant perturbations. It is reset to 1 every time the global impurity measure is improved
            if np.random.rand() <= np.exp(-stagnant):
                best_split_score = best_wm_score
                w[m] = best_wm
                stagnant += 1



        # If we've achieved a fantastic split, stop immediately
        if best_split_score < 1e-3:
            break
    #
    #
    #....................Sanity Check.........................
    #
    #
    idx = np.where(w == np.inf)[0]
    if len(idx) != 0:
        w[idx]= np.random.rand(len(idx))
    idx = np.where(w == -np.inf)[0]
    if len(idx) != 0:
        w[idx] = -1 *(np.random.rand(len(idx)))
    if b == np.inf:
        b = 10
    if b == -np.inf:
        b = -10
    #
    #
    # Now that we have a split, perform a final partition
    #
    #
    margin = np.dot(X, w) + b
    left, right = margin <= 0, margin > 0
    if (len(y[left]) == 0 ):
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    elif(len(y[right]) == 0):
        return LeafNode(is_classifier=is_classification, value=label, conf=conf,
                        samples=current_samples, features=current_features), current_depth

    else:

        # Create a decision node
        decision_node = Node(w, b, is_classifier=is_classification, value=label, conf=conf,
                               samples=current_samples, features=current_features)

        # Grow the left branch and insert it
        left_node, left_depth = build_oblique_tree_oc1(X[left, :], y[left],
                                                   is_classification, criterion,
                                                   max_depth, min_samples_split, min_features_split,
                                                   current_depth=current_depth + 1,
                                                   current_features=current_features,
                                                   current_samples=current_samples[left])
        decision_node.add_left_child(left_node)

        # Grow the right branch and insert it
        right_node, right_depth = build_oblique_tree_oc1(X[right, :], y[right],
                                                     is_classification, criterion,
                                                     max_depth, min_samples_split, min_features_split,
                                                     current_depth=current_depth + 1,
                                                     current_features=current_features,
                                                     current_samples=current_samples[right])
        decision_node.add_right_child(right_node)

        return decision_node, max(left_depth, right_depth)




# Get the best splitting threshold for each feature/attribute by considering them independently of the others
def get_best_splits(X, y, criterion=split_criteria.gini):
    n_samples, n_features = X.shape
    all_splits = np.zeros((n_features, 2))

    for f in range(n_features):
        feature_values = np.sort(X[:, f])
        feature_splits = np.convolve(feature_values, [0.5, 0.5])[1:-1]
        scores = np.empty_like(feature_splits)
        best_split = None
        best_score = np.inf

        # Compute the scores
        for i, s in enumerate(feature_splits):
            left, right = y[X[:, f] <= s], y[X[:, f] > s]
            scores[i] = criterion(left, right)
            if scores[i] < best_score:
                best_score = scores[i]
                best_split = s

        all_splits[f, :] = [best_split, best_score]

    return all_splits




