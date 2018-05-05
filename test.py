"""
sample data taken from http://cs231n.github.io/neural-networks-case-study/
neural network functions are original
"""

import numpy as np
from NN_from_scratch import simpleMLP
import pickle

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
X_test = np.zeros((N*K, D))
Y_test = np.zeros(N*K, dtype='uint8')
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

    t2 = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X_test[ix] = np.c_[r*np.sin(t2), r*np.cos(t2)]
    Y_test[ix] = j    
    

#Turn ys into 1-hot vectors
mapping_dict = {0: np.array([1,0,0]), 1: np.array([0,1,0]), 2: np.array([0,0,1])}
Y = [mapping_dict[i] for i in y]
Y_test = [mapping_dict[i] for i in Y_test]

print("X shape: %s\nY shape: %s" % (X[0].shape, Y[0].shape))
#"""
model = simpleMLP(depth=3, size=12, x_shape=X.shape[1], y_shape=Y[0].shape[0])
prev_loss = None
for epoch in range(10000):
    for sample in range(len(X)):        
        model.train_step(X[sample], Y[sample])
    loss = np.round(model.loss, decimals=3)
    if epoch % 10 == 0:
        if loss == prev_loss:
            print("converged!")
            with open("outs.pkl", 'wb') as f:
                f.write(pickle.dumps(model.weights))
                
            break
        print("epoch %s, loss: %s" % (epoch, loss))
    prev_loss = loss

correct = 0
for sample in range(len(X_test)):
    inference = model.infer(X_test[sample])
    print(X_test[sample], Y_test[sample], inference)
    if np.argmax(inference) == np.argmax(Y_test[sample]):
        correct += 1

print(correct/len(X_test))