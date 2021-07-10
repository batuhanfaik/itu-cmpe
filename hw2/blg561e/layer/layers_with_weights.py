# Adapted from Stanford CS231n Course

from .layer import Layer
from copy import copy
from abc import abstractmethod
import numpy as np


class LayerWithWeights(Layer):
    """
        Abstract class for layer with weights(CNN, Affine etc...)
    """

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


class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')

        # Calculate output's H and W according to your lecture notes
        out_H = np.int(((H + 2 * self.padding - FH) / self.stride) + 1)
        out_W = np.int(((W + 2 * self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])

        # TO DO: Do cross-correlation by using for loops
        for instance in range(N):  # for each input data
            for filter in range(F):  # per filter
                for height in range(out_H):
                    for width in range(out_W):
                        h = height * self.stride  # Starting height
                        w = width * self.stride  # Starting width
                        window = padded_x[instance, :, h:h + FH, w:w + FW]  # get the region to crop
                        # Calculate forward (x_*w[f]+b[f])
                        result = np.add(np.sum(np.multiply(window, self.W[filter])), self.b[filter])
                        out[instance, filter, height, width] = result

        return out

    def backward(self, dprev):
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)


        # Your implementation here
        padded_dx = np.pad(dx_temp, ((0, 0), (0, 0), (self.padding, self.padding),
                                     (self.padding, self.padding)), 'constant')
        for instance in range(N):  # for each input data
            for filter in range(F):  # per filter
                for height in range(out_H):
                    for width in range(out_W):
                        h = height * self.stride  # Starting height
                        w = width * self.stride  # Starting width
                        # Get regions to backprop
                        window = padded_x[instance, :, h:h + FH, w:w + FW]
                        dx_window = padded_dx[instance, :, h:h + FH, w:w + FW]
                        dprev_window = dprev[instance, filter, height, width]
                        # Calculate backprop
                        dw[filter] = np.add(dw[filter], np.multiply(window, dprev_window))
                        db[filter] = np.add(db[filter], dprev_window)
                        dx_window = np.add(dx_window, np.multiply(self.W[filter], dprev_window))
                        # Replace the values in dx
                        padded_dx[instance, :, h:h + FH, w:w + FW] = dx_window

        # Remove padding
        dx = padded_dx[:, :, self.padding:self.padding + out_H, self.padding:self.padding + out_W]

        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db
