#!/usr/bin/env python
from roc import load_data, train_test_split, plot_roc
from logreg_classifier import LogisticRegressionClassifier

def main():
    features, targets = load_data()

    logreg = LogisticRegressionClassifier()
    train_x, train_y, test_x, test_y = train_test_split(features, targets, 1/3)

    logreg.fit(train_x, train_y)
    probabilities = logreg.predict_proba(test_x)

    plot_roc(test_y, probabilities)
    
if __name__ == "__main__":
    main()

