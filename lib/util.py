import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(n):
    return np.maximum(0, n)

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def identity_function(x):
    return x

Y = identity_function(1)

print(Y)