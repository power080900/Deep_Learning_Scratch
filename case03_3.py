import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x2 = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x2))

a = np.arange(-5.0, 5.0, 0.1)
b = sigmoid(x)

plt.plot(a,b)
plt.ylim(-0.1, 1.1)
plt.show()

def relu(n):
    return np.maximum(0, n)
