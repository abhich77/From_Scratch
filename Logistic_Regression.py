import numpy as np
from sigmoid import sigmoid
from Binary_cross_entropy import BCE

class Logistic_Regression:
    def __init__(self,epochs,lr=0.01,):
        self.epochs=epochs
        self.lr=lr
        self.weights=None

    def fit(self,X,Y):
        n,m=X.shape
        self.weights=np.zeros(m)

        y_hat=sigmoid(X@self.weights)





