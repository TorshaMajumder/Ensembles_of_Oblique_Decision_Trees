#
#.............Pipeline for calling the Oblique Decision Trees using the Scikit-Learn's Bagging Classifier...............
#
#
# Importing all the oblique decision trees
#
#

from WODT import *
from HouseHolder_CART import *
from RandCART import *
from CO2 import *
from NDT import *
from Oblique_Classifier_1 import *
from DNDT import *
from segmentor import *
#
#
# Importing all the packages
#
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import load_iris, load_wine



def pre_process(dataset='breastcancer'):
    X, y = None, None

    if dataset == 'glass':
        dataset = pd.read_csv('../dataset/glass.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:9]
        y = dataset[:, 9]
        y = np.array(y, dtype=int)

    elif dataset == 'heart':
        dataset = pd.read_csv('../dataset/heart.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:13]
        y = dataset[:, 13]
        y = np.array(y, dtype=int)

    elif dataset == 'pendigits':
        dataset = pd.read_csv('../dataset/pendigits.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:16]
        y = dataset[:, 16]
        y = np.array(y, dtype=int)

    elif dataset == 'vehicle':
        dataset = pd.read_csv('../dataset/vehicle.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:18]
        y = dataset[:, 18]
        y = np.array(y, dtype=int)

    elif dataset == 'iris':
        X, y = load_iris(return_X_y=True)

    elif dataset == 'breastcancer':
        dataset = pd.read_csv('../dataset/breastcancer.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:9]
        y = dataset[:, 9]
        y = np.array(y, dtype=int)

    elif dataset == 'diabetes':
        dataset = pd.read_csv('../dataset/diabetes.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:8]
        y = dataset[:, 8]
        y = np.array(y, dtype=int)

    elif dataset == 'fourclass':
        dataset = pd.read_csv('../dataset/fourclass.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:2]
        y = dataset[:, 2]
        y = np.array(y, dtype=int)

    elif dataset == 'segment':
        dataset = pd.read_csv('../dataset/segmentation.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 1:20]
        y = dataset[:, 0]
        y = np.array(y, dtype=int)

    elif dataset == 'satimage':
        dataset = pd.read_csv('../dataset/satimage.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 0:36]
        y = dataset[:, 36]
        y = np.array(y, dtype=int)

    elif dataset == 'letter':
        dataset = pd.read_csv('../dataset/letter.csv', header=None, delimiter=',')
        dataset = np.array(dataset)
        X = dataset[:, 1:17]
        y = dataset[:, 0]
        y = np.array(y, dtype=int)

    elif dataset == 'wine':
        X, y = load_wine(return_X_y=True)
    else:
        ValueError('Unknown results set!')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    return X, y



def make_estimator(method='wodt', max_depth=5, n_estimators=10):
    if method == 'wodt':
        return WeightedObliqueDecisionTreeClassifier(max_depth=max_depth)
    elif method == 'oc1':
        return ObliqueClassifier1(max_depth=max_depth)
    elif method == 'stdt':
        return DecisionTreeClassifier(max_depth=max_depth)
    elif method == 'ndt':
        return NDTClassifier(max_depth=max_depth)
    elif method == 'wodt_bag':
        return BaggingClassifier(base_estimator=WeightedObliqueDecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'oc1_bag':
        return BaggingClassifier(base_estimator=ObliqueClassifier1(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'stdt_bag':
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'ndt_bag':
        return BaggingClassifier(base_estimator=NDTClassifier(max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'hhcart':
        return HHCartClassifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'randcart':
        return RandCARTClassifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'co2':
        return CO2Classifier(MSE(),MeanSegmentor(), max_depth = max_depth)
    elif method == 'hhcart_bag':
        return BaggingClassifier(base_estimator=HHCartClassifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'randcart_bag':
        return BaggingClassifier(base_estimator=RandCARTClassifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'co2_bag':
        return BaggingClassifier(base_estimator=CO2Classifier(MSE(),MeanSegmentor(),max_depth=max_depth), n_estimators=n_estimators)
    elif method == 'random_forest':
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    else:
        ValueError('Unknown model!')


def evaluate(datasets_to_evaluate, methods_to_evaluate):
    n_depths = 10
    n_estimators = 5
    for dataset in datasets_to_evaluate:
        print('\n--- Evaluating results set: {0}'.format(dataset))
        X, y = pre_process(dataset)
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.33, random_state=42)
        for max_depth in range(1, n_depths + 1):
            print('\n\n------ max_depth={0} \n\n'.format(max_depth), end='')

            for method in methods_to_evaluate:
                print('method={0} \n'.format(method), end='')
                if method == 'dndt':

                    transformed_Y = np.reshape(train_Y, (-1, 1))
                    onehotencoder = OneHotEncoder()
                    transformed_Y = onehotencoder.fit_transform(transformed_Y).toarray()
                    d = X.shape[1]
                    num_class = len(np.unique(y))
                    Y_pred = dndt_fit(train_X, test_X, transformed_Y, d, num_class, 1)
                    acc = np.mean(Y_pred == test_Y)
                    print("accuracy_score: ", acc,"\n")

                elif method == 'dndt_bag':

                    transformed_Y = np.reshape(train_Y, (-1, 1))
                    onehotencoder = OneHotEncoder()
                    transformed_Y = onehotencoder.fit_transform(transformed_Y).toarray()
                    d = X.shape[1]
                    num_class = len(np.unique(y))
                    Y_pred = dndt_fit(train_X, test_X, transformed_Y, d, num_class, n_estimators)
                    acc = np.mean(Y_pred == test_Y)
                    print("accuracy_score: ", acc,"\n")

                else:

                    estimator = make_estimator(method=method, max_depth=max_depth, n_estimators=n_estimators)
                    estimator.fit(train_X, train_Y)
                    Y_pred = estimator.predict(test_X)
                    acc = np.mean(Y_pred == test_Y)
                    print("accuracy_score: ",acc,"\n")



if __name__ == '__main__':
    #datasets = ['iris', 'wine', 'glass', 'heart', 'breastcancer', 'diabetes',
    # 'vehicle', 'fourclass', 'segmentation', 'satimage', 'pendigits', 'letter']
    datasets = ['iris']
    methods = ['co2_bag','oc1_bag','hhcart_bag','randcart_bag','ndt_bag','wodt_bag','dndt_bag','stdt_bag']
    evaluate(datasets, methods)


