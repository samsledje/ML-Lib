"""
Logistic Regression Classifier using gradient descent
"""
import numpy as np
import math

class LogisticRegressionClassifier:
    def __init__(self):
        """
        For n training examples
            x: n*m matrix of training examples
            w: 1*m vector of weights
            y: n*1 column vector of labels
        """
        self.w = None
        self.alpha = 0.001
        self.epsilon = 0.001
        self.n = 0
        self.m = 0

    def fit(self, x, y):
        """Learns weights using gradient descent
        
        Arguments:
            x {np.ndarray} -- training examples
            y {np.ndarray} -- labels
        """

        self.n, self.m = x.shape
        self.w = np.zeros(x.shape[1])
        epsilon = float("inf")
        iters = 0
        while self.epsilon < epsilon:
            ll_pre = self._log_likelihood(self.w, x, y)
            self._update_weights(x, y)
            ll_new = self._log_likelihood(self.w, x, y)
            epsilon = abs(ll_pre - ll_new)
            iters += 1

    def predict(self, x):
        """Predicts labels given test examples
        
        Arguments:
            x {np.ndarray} -- test data
        
        Returns:
            np.ndarray -- predicted labels
        """

        y = np.zeros(self.n)
        proba = self.predict_proba(x)
        for i in range(self.n):
            y[i] = 1 if proba[i] > 0.5 else 0
        return y

    def predict_proba(self, x):
        """Predicts labels given test examples
        
        Arguments:
            x {np.ndarray} -- test data
        
        Returns:
            np.ndarray -- probability of class membership
        """
        assert x.shape[1] == (self.m), "X has improper dimensions"

        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = math.exp(np.dot(x[i].transpose(), self.w))/(1+math.exp(np.dot(x[i].transpose(), self.w)))
        return y


    def _log_likelihood(self, w, x, y):
        """Computes log likelihood 
        
        Arguments:
            w {np.ndarray} -- weights
            x {np.ndarray} -- training examples
            y {np.ndarray} -- labels
        """

        ll = np.zeros(self.n)
        for i in range(self.n):
            ll[i] = np.dot(y[i], np.dot(w.transpose(), x[i])) - math.log(1 + math.exp(np.dot(w.transpose(), x[i])))
        return(np.sum(ll))

    def _update_weights(self, x, y):
        grad_matrix = np.zeros((self.n, self.m))
        for i in range(self.n):
            first_term = np.dot(y[i], x[i])
            second_term = np.dot(x[i], math.exp(np.dot(self.w.transpose(), x[i])))
            third_term = (1 + math.exp(np.dot(self.w.transpose(), x[i])))
            grad_matrix[i] = first_term - (second_term / third_term)
        grad = np.sum(grad_matrix, axis=0)
        self.w = self.w - (self.epsilon * grad)
