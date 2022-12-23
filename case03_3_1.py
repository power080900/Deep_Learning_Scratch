import numpy as np


A = np.array([1,2,3,4])
np.ndim(A)
A.shape[0]

B = np.array([[1,2,3],[4,5,6]])
np.ndim(B)
B.shape

A = np.array([[1,2],[3,4]])
np.dot(A,B)

X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])

Y = np.dot(X,W)
print(Y)