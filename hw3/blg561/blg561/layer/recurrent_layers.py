# Adapted from CS231n
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
        H = h0.shape[1]
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

        dprev_h_t = np.zeros((N, H))
        dx = np.zeros((N, T, D))
        dWh = np.zeros((H, H))
        dWx = np.zeros((D, H))
        db = np.zeros(H)
        r_time = np.arange(T - 1, -1, -1)  # Backprop through time

        for batch in r_time:
            dx_t, dprev_h_t, dWx_t, dWh_t, db_t = self.backward_step(dh[:, batch, :] + dprev_h_t,
                                                                     self.cache[batch])
            dx[:, batch, :] = dx_t
            dWh += dWh_t
            dWx += dWx_t
            db += db_t

        dh0 = dprev_h_t
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
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

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
        H = prev_h.shape[1]
        a = self.b + np.dot(prev_h, self.Wh) + np.dot(x, self.Wx)

        # Using np.split() works too, this is just more clear so I can debug easier
        a_i = a[:, 0 * H:1 * H]
        a_f = a[:, 1 * H:2 * H]
        a_o = a[:, 2 * H:3 * H]
        a_g = a[:, 3 * H:4 * H]

        i = self.sigmoid(a_i)
        f = self.sigmoid(a_f)
        o = self.sigmoid(a_o)
        g = np.tanh(a_g)

        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (prev_h, prev_c, next_h, next_c, x, i, f, o, g, a)

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
        N, T, D = x.shape
        H = h0.shape[1]
        h = np.empty((N, T, H))
        prev_c = np.zeros((N, H))
        time = np.arange(T)
        cache = []

        prev_h = h0
        for batch in time:
            next_h, next_c, cache_ = self.forward_step(x[:, batch, :], prev_h, prev_c)
            cache.append(cache_)
            h[:, batch, :] = next_h
            prev_h = next_h
            prev_c = next_c

        self.cache = cache.copy()
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
        # Note that grad of sigmoid is not \sigma * (1-\sigma) because passed in parameters in the
        # cache are in sigmoid(x) form
        grad_sigmoid = lambda x: x * (1 - x)
        prev_h, prev_c, next_h, next_c, x, i, f, o, g, a = cache
        H = prev_h.shape[1]

        do = dnext_h * np.tanh(next_c) * grad_sigmoid(o)

        dc = dnext_c + dnext_h * o * (1 - np.power(np.tanh(next_c), 2))
        dprev_c = dc * f

        df = dc * prev_c * grad_sigmoid(f)
        di = dc * g * grad_sigmoid(i)
        dg = dc * i * (1 - np.power(g, 2))

        # np.concatenate() on columns or np.hstack() would work just as fine
        da = np.empty(a.shape)
        da[:, 0 * H:1 * H] = di
        da[:, 1 * H:2 * H] = df
        da[:, 2 * H:3 * H] = do
        da[:, 3 * H:4 * H] = dg

        dprev_h = np.dot(da, np.transpose(self.Wh))
        dx = np.dot(da, np.transpose(self.Wx))
        dWh = np.dot(np.transpose(prev_h), da)
        dWx = np.dot(np.transpose(x), da)
        db = np.sum(da, axis=0)

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
        N, T, H = dh.shape
        try:
            D = self.cache[0][4].shape[1]
        except IndexError:
            D = None
            assert "Cache is not valid"

        dnext_h_t = np.zeros((N, H))
        dnext_c_t = np.zeros((N, H))
        dx = np.zeros((N, T, D))
        dWh = np.zeros((H, 4 * H))
        dWx = np.zeros((D, 4 * H))
        db = np.zeros(4 * H)
        r_time = np.arange(T - 1, -1, -1)  # Backprop through time

        for batch in r_time:
            dx_t, dnext_h_t, dnext_c_t, dWx_t, dWh_t, db_t = self.backward_step(
                dh[:, batch, :] + dnext_h_t, dnext_c_t, self.cache[batch])
            dx[:, batch, :] = dx_t
            dWh += dWh_t
            dWx += dWx_t
            db += db_t

        dh0 = dnext_h_t
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
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        H = prev_h.shape[1]
        a = self.b + np.dot(prev_h, self.Wh) + np.dot(x, self.Wx)

        # Using np.split() works too, this is just more clear so I can debug easier
        a_z = a[:, 0 * H:1 * H]
        a_r = a[:, 1 * H:2 * H]

        z = self.sigmoid(a_z)
        r = self.sigmoid(a_r)

        h_candidate = np.tanh(self.bi + np.dot(r * prev_h, self.Whi) + np.dot(x, self.Wxi))
        next_h = z * prev_h + (1 - z) * h_candidate
        cache = (prev_h, next_h, x, z, r, a, h_candidate)
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
        H = h0.shape[1]
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
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWi: gradients of weights Wxi, of shape (D, H)
            dWhi: gradients of weights Whi, of shape (H, H)
            dbi: gradients of bias bi, of shape (H,)
        """
        # Note that grad of sigmoid is not \sigma * (1-\sigma) because passed in parameters in the
        # cache are in sigmoid(x) form
        grad_sigmoid = lambda x: x * (1 - x)
        prev_h, next_h, x, z, r, a, h_candidate = cache
        H = prev_h.shape[1]

        da_z = (dnext_h * (h_candidate - prev_h) + 1) * grad_sigmoid(z)
        dai = dnext_h * z * (1 - np.power(np.tanh(h_candidate), 2))
        da_r = np.dot(dai, np.transpose(self.Whi)) * prev_h * grad_sigmoid(r)

        # np.concatenate() on columns or np.hstack() would work just as fine
        da = np.empty(a.shape)
        da[:, 0 * H:1 * H] = da_z
        da[:, 1 * H:2 * H] = da_r

        dprev_h = dnext_h * (1 - z) + np.dot(da, np.transpose(self.Wh))\
                + r * np.dot(dai, np.transpose(self.Whi))
        dx = np.dot(da, np.transpose(self.Wx)) + np.dot(dai, np.transpose(self.Wxi))
        dWh = np.dot(np.transpose(prev_h), da)
        dWx = np.dot(np.transpose(x), da)
        db = np.sum(da, axis=0)
        dWhi = np.dot(np.transpose(prev_h * r), dai)
        dWxi = np.dot(np.transpose(x), dai)
        dbi = np.sum(dai, axis=0)

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
        N, T, H = dh.shape
        try:
            D = self.cache[0][4].shape[1]
        except IndexError:
            D = None
            assert "Cache is not valid"

        dnext_h_t = np.zeros((N, H))
        dx = np.zeros((N, T, D))
        dWh = np.zeros((H, 2 * H))
        dWx = np.zeros((D, 2 * H))
        db = np.zeros(2 * H)
        dWhi = np.zeros((H, H))
        dWxi = np.zeros((D, H))
        dbi = np.zeros(H)
        r_time = np.arange(T - 1, -1, -1)  # Backprop through time

        for batch in r_time:
            dx_t, dnext_h_t, dWx_t, dWh_t, db_t, dWxi_t, dWhi_t, dbi_t = self.backward_step(
                dh[:, batch, :] + dnext_h_t, self.cache[batch])
            dx[:, batch, :] = dx_t
            dWh += dWh_t
            dWx += dWx_t
            db += db_t
            dWhi += dWhi_t
            dWxi += dWxi_t
            dbi += dbi_t

        dh0 = dnext_h_t
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db, 'dWxi': dWxi,
                     'dWhi': dWhi, 'dbi': dbi}
