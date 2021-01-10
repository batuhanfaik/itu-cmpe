#Adapted from CS231n
from .layers_with_weights import LayerWithWeights
from copy import deepcopy
from abc import abstractmethod
import numpy as np


class RNNLayer(LayerWithWeights):
    """ Simple RNN Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ RNN Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, out_size)
        self.Wh = np.random.rand(out_size, out_size)
        self.b = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None, 'dWh': None, 'db': None}
        
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        next_h = np.tanh(self.b + np.dot(prev_h, self.Wh) + np.dot(x, self.Wx))
        cache = (prev_h, next_h, x)
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        _, H = h0.shape
        h = np.empty((N, T, H))
        time = np.arange(T)
        cache = []

        prev_h = h0
        for batch in time:
            next_h, cache_ = self.forward_step(x[:, batch, :], prev_h)
            cache.append(cache_)
            h[:, batch, :] = next_h
            prev_h = next_h

        self.cache = cache.copy()
        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
        """
        prev_h, next_h, x = cache
        dtanh = dnext_h * (1 - np.power(next_h, 2))
        dprev_h = np.dot(dtanh, np.transpose(self.Wh))
        dx = np.dot(dtanh, np.transpose(self.Wx))
        dWh = np.dot(np.transpose(prev_h), dtanh)
        dWx = np.dot(np.transpose(x), dtanh)
        db = np.sum(dtanh, axis=0)

        return dx, dprev_h, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
            }
        """
        N, T, H = dh.shape
        try:
            D = self.cache[0][2].shape[1]
        except IndexError:
            D = None
            assert "Cache is not valid"

        dprev_t = np.zeros((N, H))
        dx = np.zeros((N, T, D))
        dWh = np.zeros((H, H))
        dWx = np.zeros((D, H))
        db = np.zeros(H)
        r_time = np.arange(T-1, -1, -1)    # Backprop through time

        for batch in r_time:
            dx_t, dprev_t, dWx_t, dWh_t, db_t = self.backward_step(dh[:, batch, :] + dprev_t,
                                                                   self.cache[batch])
            dx[:, batch, :] = dx_t
            dWh += dWh_t
            dWx += dWx_t
            db += db_t

        dh0 = dprev_t
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db}
        
        
class LSTMLayer(LayerWithWeights):
    """ Simple LSTM Layer - only calculates hidden states and cell states """
    def __init__(self, in_size, out_size):
        """ LSTM Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 4 * out_size)
        self.Wh = np.random.rand(out_size, 4 * out_size)
        self.b = np.random.rand(4 * out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None}
        
    def forward_step(self, x, prev_h, prev_c):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
            prev_c: previous cell state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            next_c: next cell state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return next_h, next_c, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return h
        
    def backward_step(self, dnext_h, dnext_c, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            dnext_c: gradient of loss with respect to
                     cell state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dprev_c: gradients of previous cell state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
            }
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db}
        

class GRULayer(LayerWithWeights):
    """ Simple GRU Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ GRU Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 2 * out_size)
        self.Wh = np.random.rand(out_size, 2 * out_size)
        self.b = np.random.rand(2 * out_size)
        self.Wxi = np.random.rand(in_size, out_size)
        self.Whi = np.random.rand(out_size, out_size)
        self.bi = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None, 'dWxi': None,
                     'dWhi': None, 'dbi': None}
        
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWi: gradients of weights Wxi, of shape (D, H)
            dWhi: gradients of weights Whi, of shape (H, H)
            dbi: gradients of bias bi, of shape (H,)
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        return dx, dprev_h, dWx, dWh, db, dWxi, dWhi, dbi

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWxi: gradients of weights Wx, of shape (D, H)
            dWhi: gradients of weights Wh, of shape (H, H)
            dbi: gradients of bias b, of shape (H,)
            }
        """
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        raise NotImplementedError
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db, 'dWxi': dWxi, 'dWhi': dWhi, 'dbi': dbi}

