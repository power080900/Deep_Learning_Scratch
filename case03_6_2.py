import numpy as np
import dataset.util as utl

x, t = utl.get_data()
network = utl.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = utl.predict(network , x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:"+ str(float(accuracy_cnt) / len(x)))