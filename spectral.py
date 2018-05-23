import sys
import numpy as np
from numpy import linalg as la
from utils import *

M = [   [0,         .7074,      .1198,      .1962,      .6065],
        [.7074,     0,          .0153,      .0330,      .8574],
        [.1198,     .0153,      0,          .7533,      .0121],
        [.1962,     .0330,      .7533,      0,          .0377],
        [.6065,     .8574,      .0121,      .0377,      0    ]
    ]

A = np.asarray(M)
print("A")
print(A)

# calculate D
D = np.zeros(A.shape)
for i in range(A.shape[0]):
        D[i][i] = A[i].sum(axis=0)
print("\nD")
print(D)

L = D - A
print("\nL = D - A")
print(L)

values, vectors = la.eig(L)
print("\nSecond Eigenvector")
print(vectors[2])

rounded = np.empty((vectors[2].shape))
for i in range(vectors[2].shape[0]):
    if vectors[2][i] > 0:
        rounded[i] = 1
    else:
        rounded[i] = 0
print("\nClustering")
print(rounded)
