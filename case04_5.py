
import sys, os
sys.path.append(os.pardir)
from dataset.util import *

class TwoLauyerNet:
    def __init__(self, input_size, hidden_size, output_size, weiht_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weiht_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weiht_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 =  self.params['W1'], self.params['W2']
        