import sys
import numpy as np
from numpy import linalg as la
from utils import *

M = [   [0,1,0,0,1,0],
        [1,0,1,0,1,0],
        [0,1,0,1,0,0],
        [0,0,1,0,1,1],
        [1,1,0,1,0,0],
        [0,0,0,1,0,0]
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
