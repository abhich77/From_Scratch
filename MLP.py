import numpy as np

class MLP:
    def __init__(self,input_size,hidden_size,output_size):
        self.weights_input_hidden=np.zeros(input_size,hidden_size)
        self.weights_hidden_output=np.zeros(hidden_size,output_size)
        self.bias_hidden=np.zeros(1,hidden_size)
        self.bias_output=np.zeros(1,output_size)

        

    def forward(self,X,binary=True):
        self.hidden_input=np.dot(X,self.weights_input_hidden)+self.bias_hidden
        self.hidden_output=self.sigmoid(self.hidden_output)
        self.final_input=np.dot(self.hidden_output,self.weights_hidden_output)+self.bias_output
        self.final_output=self.sigmoid(self.final_output) if binary else self.final_output=self.softmax(self.final_output)
        return self.final_output

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def softmax(self,X):
        #this is a modified  version
        exp_x=np.exp(X-np.max(X))
        return exp_x/exp_x.sum(axis=-1,keepdims=True)

    def backward(self,X,Y,l_rate,output):
        output_error=output-Y
        self.hidden_error=np.dot(output_error,self.weights_hidden_output.T)*self.hidden_output*(1-self.hidden_output)

        self.weights_hidden_output-=l_rate*np.dot(self.hidden_error.T,output_error)
        self.bias_output-=l_rate*np.sum(output_error,axis=0,keepdims=True)
        self.weights_input_hidden-=l_rate*np.dot(X.T,self.hidden_output)
        self.bias_hidden-=l_rate*np.sum(self.hidden_error,axis=0,keepdims=True)

    def fit(self,X,Y,epochs,l_rate):
        for ep in range(epochs):
            output=self.forward(X)
            self.backward(X,Y,l_rate,output)
            if (ep+1) % 100 == 0:
                loss = -np.sum(y * np.log(output)) / X.shape[0]
                print(f'Epoch {ep+1}, Loss: {loss:.4f}')


    




