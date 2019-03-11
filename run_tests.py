#!/usr/bin/env python3
# Basic imports
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# sklearn Imports
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.metrics import mean_squared_error, r2_score

# SaML Imports
from Regression import LinearRegression, LogisticRegressionClassifier
from Clustering import KMeansClustering, AggHierClustering
from Plotting import plot_clusters, plot_linear, plot_roc, plot_sse
from utils import print_header, read_data, backspace, write, train_test_split

# Path to Datasets
DATA_PATH = ('test_data/{}')

def test_linear_regression():
    """Taken from http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    """
    print_header("Linear Regression on sklearn diabetes dataset")

    diabetes = load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create and fit classifier
    LRClass = LinearRegression()
    LRClass.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions
    diabetes_y_pred = LRClass.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', LRClass._weights)
    # The mean squared error
    print("Mean squared error: %.2f"
        % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plot_linear(diabetes_X_test, diabetes_y_pred)

def test_logistic_regression():
    features,targets = load_breast_cancer(True)

    logreg = LogisticRegressionClassifier()
    train_x, train_y, test_x, test_y = train_test_split(features, targets, 1/3)

    logreg.fit(train_x, train_y)
    probabilities = logreg.predict_proba(test_x)
    plot_roc("ROC Curve for Wisconsin Breast Cancer Dataset", test_y, probabilities)

def test_k_means():
    from Clustering import calculate_sse
    data = read_data(DATA_PATH.format("k_means_data"))

    print_header("SSE with varying k")

    k_means_sse = {}
    for k in range(2, 11):
        k_means = KMeansClustering(data, k, large_output=True)
        sse = calculate_sse(k_means[1], k_means[2])
        k_means_sse[k] = sse
        print("SSE for k={} is {}".format(k, sse))
    plot_sse(k_means_sse)

    print_header("Clustering with k = 3")

    k_3_clustering = KMeansClustering(data, 3)
    plot_clusters(k_3_clustering)

def test_agg_hier():
    data = read_data(DATA_PATH.format("agg_hier_data"))
    method = 'min'

    print_header("Agglogmerative Hierarchical Clustering with {} and k = 2".format(method))

    agg_hier_clustering = AggHierClustering(data, 2, method)
    plot_clusters(agg_hier_clustering)

if __name__ == "__main__":
    print_header("Select Algorithm to Test")
    case = input(   "1: Linear Regression\n"+
                    "2: Logistic Regression\n"+
                    "3: K Means Clustering\n"+
                    "4: Agglomerative Hierarchical Testing\n")
    case = int(float(case.strip()))
    if case == 1:
        test_linear_regression()
    elif case == 2:
        test_logistic_regression()
    elif case == 3:
        test_k_means()
    elif case == 4:
        test_agg_hier()
    else:
        print("Invalid input")
        quit(1)
