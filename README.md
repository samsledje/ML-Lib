# Logistic Regression Classifier

Logistic Regression Classifier using gradient descent

Written for UConn CSE 5820 (Machine Learning)

- logreg\_classifier.py contains the Logistic Regression Classifier

- roc.py contains function for plotting ROC.

- main.py loads Wisconsin Breast Cancer dataset from sklearn (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) and divides it into a training set and test set. It then fits the logistic regression classifier using the training set, predicts probabilities on the test set, and plots the ROC curve

- Runtime.ipynb is a Jupyter notebook which performs the same functions as main.py, then fits an sklearn logistic regression classifier and plots an ROC curve on the same datasets.
