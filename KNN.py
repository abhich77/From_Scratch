import numpy as np

class KNN:
    def __init__(self,k=2,metric='euclidean',p=2):
        self.k=k
        self.metric=metric
        self.p=p

    def euclidean(self,v1,v2):
        return np.sqrt(np.sum((v1-v2)**2))


    def manhattan(self,v1,v2):
        return np.sum(np.abs(v1-v2))

    def minkowski(self,v1,v2,p=2):
        return np.sum(np.abs(v1-v2)**p)**(1/p)

    def fit(self,X,Y):
        self.X=X
        self.Y=Y

    def get_neighbour(self,data):
        distances=list()

        for (x,y) in zip(self.X,self.Y):
            if self.metric=='euclidean':
                dist=self.euclidean(x,data)

            elif self.metric=='manhattan':
                dist=self.manhattan(x,data)

            elif self.metric=='minkowski':
                dist=self.minkowski(x,data,self.p)
            
            distances.append((dist,y))
        distances.sort(key=lambda x:x[0])

        neighbours=list()

        for i in range(self.k):
            neighbours.append(distances[i][1])

        return neighbours

    def predict(self,X):
        preds=[]

        for x in X:
            neighbour=self.get_neighbour(x)

            majority=np.bincount(neighbour).argmax()
            preds.append(majority)
        return np.array(preds)
            

