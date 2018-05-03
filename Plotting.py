import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_clusters(data):
    x, y, cluster = zip(*data)
    plt.scatter(x, y, c = cluster, cmap=cm.Dark2)
    plt.title("Clustering, k = {}".format(len(np.unique(cluster))))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def plot_sse(data):
    k, sse = zip(*data.items())
    plt.plot(k, sse)
    plt.title("Sum of Squared Error")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.show()

def plot_linear(x, y):
    # Plot outputs
    plt.plot(x, y, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def plot_roc(title, label, proba):
    """Plots ROC curve

    Arguments:
        label {np.ndarray} -- true label
        proba {np.ndarray} -- predicted probabilites of class membership
    """
    rates = []

    pair = pd.concat((pd.DataFrame(proba), pd.DataFrame(label)), axis=1)
    pair.columns = ("proba", "target")
    pair_sorted = pair.sort_values(pair.columns[0], ascending=False)
    matrix = pair_sorted.as_matrix()
    for i in matrix:
        thresh = i[0]
        rates.append(_calc_rates(matrix, thresh))

    r = pd.DataFrame(rates)
    plt.plot(r[1].values, r[0].values)
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0,1.1, 0.1))
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive')
    plt.show()

# Helper Functions

def _calc_rates(matrix, thresh):
    """Calculates true positive rate and false positive rate for a given threshold

    Arguments:
        matrix {np.ndarray} -- true labels and predicted probabilities
        thresh {float} -- threshold for a round of ROC construction

    Returns:
        [float, float] -- true positive rate, false positive rate
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    n = len(matrix)
    for i in matrix:
        pred = 1 if i[0] >= thresh else 0
        if pred == 1:
            if i[1] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if i[1] == 0:
                tn += 1
            else:
                fn += 1
    tpr = tp / (tp+fn)
    fpr = fp / (fp + tn)
    return tpr, fpr