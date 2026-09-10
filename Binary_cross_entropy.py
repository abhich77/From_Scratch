import numpy as np

def BCE(self,y_hat,y,ep=1e-7):
    y_hat = np.clip(y_hat, ep, 1-ep)
    loss=-(1/len(y))*np.sum((y*np.log(y_hat))+((1-y)*np.log(1-y_hat)))
    return loss