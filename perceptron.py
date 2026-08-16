import numpy as np


class Perceptron:
    def __init__(self,l_rate=0.1,epochs=10):
        self.epochs=epochs
        self.l_rate=l_rate
        self.wei=None
        self.bias=None

    def fit(self,X,Y):
        n_samples,n_features=X.shape

        self.wei=np.zeros(n_features)
    
        self.bias=0

        for epoch in range(self.epochs):

            errors=0
            indices=np.random.permutation(n_samples)

            X_shuffled=X[indices]
            Y_shuffled=Y[indices]

            for x_i,y_i in zip(X_shuffled,Y_shuffled):

               out=np.dot(x_i,self.wei)+self.bias
              
               y_pred=self._step(out)

               if y_i!=y_pred:
                    update=self.l_rate*(y_i-y_pred)
                    self.wei+=update*x_i
                    self.bias+=update
                    errors+=1

            if errors==0:
                print(f"Training stopped early at epoch {epoch}")
                break

    def pred(self,X):

        out=np.dot(X,self.wei)+self.bias
        y_pred=self._step(out)
        return y_pred

    def _step(self,x):
        return np.where(x>=0,1,0)

                    

        

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y= np.array([0, 0, 0, 1])

p=Perceptron()
p.fit(X,Y)
print("Predictions:", p.pred(X))