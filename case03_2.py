import numpy as np

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
y1 = y.astype(np.int)
print(x,y,y1)