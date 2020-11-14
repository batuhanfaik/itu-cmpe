import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None
        self.w = None
        self.b = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        # This is used for avoiding the issues related to mutability of python arrays
        x = x.copy()

        # Implement relu activation
        x = x.clip(min=0)   # I am using the clip() method here,
        # because according to benchmarks it is faster than traditional methods
        return x

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        dx = None
        # Your implementation starts
        dx = np.multiply((self.x > 0), dprev)   # I am using the numpy multiplication method here,
        # as it is on average 35-50% faster than commonly used indexing and checking the value method
        # End of your implementation
        return dx


class YourActivation(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            :param x: outputs of previous layer
            :return: output of activation
        '''
        # Lets have an activation of X^2
        # TODO: CHANGE IT
        self.x = x.copy()
        out = x ** 2
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        # TODO: CHANGE IT
        # Example: derivate of X^2 is 2X
        dx = dprev * self.x * 2
        return dx


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass
        probs = None
       
        # Your implementation starts
        # Mathematical formula is given as f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}
        # (https://cs231n.github.io/linear-classify/#softmax)
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.probs = probs.copy()
        # End of your implementation

        return probs

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        dx = None
        # Your implementation starts
        num_inputs = y.shape[0]
        dx = self.probs
        dx[np.arange(num_inputs), y] -= 1
        dx /= num_inputs
        # End of your implementation

        return dx


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None
    # Your implementation starts
    # Cross entropy loss is defined below
    num_inputs = y.shape[0]
    p_i = probs[np.arange(num_inputs), y]
    loss = -np.sum(np.log(p_i)) / num_inputs
    # End of your implementation
    return loss


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        # Vectorize the input to [batchsize, others] array
        batch_size = x.shape[0]
        vectorized_x = x.reshape(batch_size, -1)
        # Do the affine transform
        out = np.dot(vectorized_x, self.W) + self.b

        # Save x for using in backward pass
        self.x = x.copy()

        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        # Vectorize the input to a 1D ndarray
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        x_vectorized = self.x.reshape(batch_size, -1)
        x_vectorized_transpose = np.transpose(x_vectorized)
        W_transpose = np.transpose(self.W)
        dx = np.dot(dprev, W_transpose)
        # Match shapes of x and W (CS231N 2017 - Lecture 4 slide 70)
        dx = dx.reshape(self.x.shape)
        dw = np.dot(x_vectorized_transpose, dprev)
        db = np.sum(dprev, axis=0)
        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'

class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)

        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

class VanillaSDGOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
         # Your implementation starts
        optim_W = m.W.copy()
        l2_term = np.sum(np.square(optim_W))
        optim_W += - self.lr * m.dW + self.reg * l2_term
        print("L2: {}, W: {}".format(l2_term, optim_W))
        return optim_W
        # End of your implementation
       
class SGDWithMomentum(VanillaSDGOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        # Save velocities for each model in a dict and use them when needed.
        # Modules can be hashed
        self.velocities = {m: 0 for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        pass
        # End of your implementation
