import numpy as np
import dataset.util as utl

network = utl.init_network()
x = np.array([1.0,0.5])
y = utl.forward(network,x)
print(y)
