import numpy as np

class KMeans:
    def __init__(self,clusters=2,epochs=10):
        self.clusters=clusters
        self.epochs=epochs

    def euclidean(self,v1,v2):
        return np.sqrt(np.sum((v1-v2)**2))

    def fit(self,X):
        m,n=X.shape
        self.centroid=X[np.random.choice(m,self.clusters,replace=False)]
        for _ in range(self.epochs):
            dist=[]
            labels=[]
            for x in X:
                dist.append([self.euclidean(x,c) for c in self.centroid])
                label=np.argmin(dist[-1])
                labels.append(label)

            labels=np.array(labels)

            for  i in range(self.clusters):
                if np.any(labels==i):
                    self.centroid[i]=np.mean(X[labels==i],axis=0)

        final_labels=[]
        for x in X:
            dist=[self.euclidean(x,c) for c in self.centroid]
            final_labels.append(np.argmin(dist))

        return np.array(final_labels),self.centroid

    def predict(self,X):
        labels=[]
        for x in X:
            dist=[self.euclidean(x,c) for c in self.centroid]
            labels.append(np.argmin(dist))

        return np.array(labels)
            