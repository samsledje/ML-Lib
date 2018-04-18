import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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