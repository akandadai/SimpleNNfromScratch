import numpy as np

"""
TODO: Explicitly calculate loss (check loss calulation) [DONE - used simple log]
Reshape arrays so that matrix multiplication works -- struggling

"""

class simpleMLP(object):
    
    def __init__(self, depth, size, x_shape, y_shape):
        
        #initialize model parameters
       
        
        self.depth = depth
        self.size = size
        self.weights = {(i+1):[0.1*np.random.randn(size, size), np.zeros((size,1))] for i in range(1,depth-1)}
        self.input_size=None
        self.output_size=None
        self.input_size = x_shape
        self.output_size = y_shape
        self.weights.update({1: [0.1*np.random.randn(size, self.input_size), np.zeros((self.size,1))],
                                 self.depth: 0.1*np.random.randn(self.output_size, self.size)})        
    def forward_pass(self, x):
        #reshape input to (input_size, 1)
        x= x.reshape((self.input_size, 1))
        #test = np.array([1,2,3,4]).reshape((4,1))
        
        self.interm_outs = {i: None for i in range(1, self.depth)}
        #append input for use during backprop
        self.interm_outs.update({0: x})
        #forward pass
        
        #initialize first layer as inputs
        previous_lr = x
        for layer in range(1,len(self.weights)):
            lr = np.tanh(np.dot(self.weights[layer][0], previous_lr) + self.weights[layer][1])
            self.interm_outs[layer] = lr
            previous_lr = lr
        
        
        #output logits
        logits = np.dot(self.weights[self.depth],self.interm_outs[self.depth - 1])
        exp_logits = np.exp(logits)
        
        probs = exp_logits/np.sum(exp_logits)
        
        
        return probs
    
    def infer(self, x):
        probs = self.forward_pass(x)
        y = np.zeros(self.output_size)
        inference = probs.argmax()
        y[inference] = 1
        return y
    
    def train_step(self, x,y):
        
        #perform forward pass
        probs = self.forward_pass(x)
        y_actual = y.argmax()
        
        #define loss = -log(prob_correctClass)
        self.loss = -np.log(probs[y_actual])
        
        #see gradient formula for softmax
        probs[y_actual] -= 1
        
        #first gradient, begin backprop
        d_logits = -probs*probs[y_actual]
        #gradient for output layer weights
        d_outLayer = np.dot(d_logits,self.interm_outs[self.depth - 1].T)
        
        #keep track of grads (for training visibility)
        self.grads = {i: None for i in range(1, self.depth + 1)}
        self.grads[self.depth] = d_outLayer
        
        #set the first gradient up to the current point
        #we will iteratively update this to mean "the gradient with respect to the layer before the current"
        d_prev = np.dot(self.weights[self.depth].T, d_logits)
        
        #Backprop to all layers
        for layer in range(self.depth - 1, 0, -1):
            d_tanh = 1 - np.square(self.interm_outs[layer])
            d_Ti = d_prev * d_tanh
            d_prev = np.dot(self.weights[layer][0].T, d_Ti)
            self.grads[layer] = [np.dot(d_Ti,self.interm_outs[layer-1].T), d_Ti]
        
        
        #parameter update
        step_size = 0.05
        self.weights[self.depth] -= step_size * self.grads[self.depth]
        for grad in range(len(self.grads)-1, 0, -1):
            self.weights[grad][0] -= step_size*self.grads[grad][0]
            self.weights[grad][1] -= step_size*self.grads[grad][1]
        
