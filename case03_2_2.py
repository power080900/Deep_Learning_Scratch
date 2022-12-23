import numpy as np
import matplotlib.pylab as plt
from lib.util import sigmoid
from lib.util import step_function

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

x2 = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x2))

a = np.arange(-5.0, 5.0, 0.1)
b = sigmoid(x)

plt.plot(a,b)
plt.ylim(-0.1, 1.1)
plt.show()


