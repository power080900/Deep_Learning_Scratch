import numpy as np

def function_2(x):
    return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 *h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))

print(numerical_gradient(function_2, np.array([0.0, 2.0])))

print(numerical_gradient(function_2, np.array([3.0, 0.0])))

def gradient_descent(f, init_x, lr= 0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad

    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.util import softmax, cross_entropy_error, numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6,0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x,t))

def f(W):
    return net.loss(x,t)

print(f)
print(net.W)
dW = numerical_gradient(net.loss(x,t), net.W)
print(dW)