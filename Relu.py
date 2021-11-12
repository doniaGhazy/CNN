import numpy as np

class Relu_Layer():
    def __init__(self):
        pass
    def name (self):
        print("Relu")
    def forward(self, X):
        self.input = X
        out = np.maximum(0, X)
        return out
    
    def backward(self, d):
        X = self.input
        out = d
        out[X <= 0] = 0
        return out
