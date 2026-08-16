import numpy as np

class LinearRegression:

    def __init__(self):
        self.m=None
        self.b=None

    def fit(self,X,Y):
        num=0
        den=0
        x_mean=X.mean()
        y_mean=Y.mean()
        # for i in range (X.shape[0]):
            # num+=(X[i]-X.mean())*(Y[i]-Y.mean())
            # den+=(X[i]-X.mean())**2
        num=np.sum((X-x_mean)*(Y-y_mean))
        den=np.sum((X-x_mean)**2)
            
        self.m=num/den

        self.b=(y_mean-self.m*x_mean)

        print(self.m)
        print(self.b)
    
    def predict(self,X):
        return self.m*X +self.b



