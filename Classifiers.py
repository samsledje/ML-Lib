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
        self.alpha = 1
        self.epsilon = 0.01
        self.n = 0
        self.m = 0
        self.ll = -float("inf")

    def fit(self, x, y):
        """Learns weights using gradient descent
        
        Arguments:
            x {np.ndarray} -- training examples
            y {np.ndarray} -- labels
        """

        self.n, self.m = x.shape
        self.w = np.ones(x.shape[1])
        epsilon = float("inf")
        scale = np.linalg.norm(x)
        y_new = y / scale
        x_new = x / scale
        self.ll = self._log_likelihood(self.w, x_new, y_new)

        while self.epsilon < epsilon:
            self._update_weights(x_new, y)
            ll_new = self._log_likelihood(self.w, x_new, y_new)
            epsilon = abs(ll_new - self.ll)
            self.ll = ll_new
            #input()
        self.w = self.w * scale

    def predict(self, x):
        """Predicts labels given test examples
        
        Arguments:
            x {np.ndarray} -- test data
        
        Returns:
            np.ndarray -- predicted labels
        """

        y = np.empty(x.shape[0])
        proba = self.predict_proba(x)
        for i in range(x.shape[0]):
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

        scale = np.linalg.norm(x)
        x_new = x / scale
        w_new = self.w / scale

        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = math.exp(np.dot(x_new[i].transpose(), w_new))/(1+math.exp(np.dot(x_new[i].transpose(), w_new)))
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
            grad_matrix[i] = np.dot(y[i], x[i]) - (np.dot(x[i], math.exp(np.dot(self.w.transpose(), x[i]))) / (1 + math.exp(np.dot(self.w.transpose(), x[i]))))
        grad = np.sum(grad_matrix, axis=0)
        self.w = self.w + (self.alpha * grad)
