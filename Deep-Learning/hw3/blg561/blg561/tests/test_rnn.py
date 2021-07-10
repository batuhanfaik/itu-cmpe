import unittest
import numpy as np

from blg561.layer.recurrent_layers import RNNLayer
from blg561.checks import rel_error, grad_check

class TestRNN(unittest.TestCase):

    def test_forward_step(self):
        N, D, H = 3, 10, 4
        rnn = RNNLayer(10, 4)
        x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
        rnn.Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
        rnn.Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
        rnn.b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn.forward_step(x, prev_h)
        expected_next_h = np.array([
          [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
          [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
          [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

        assert rel_error(expected_next_h, next_h) < 1e-6

    def test_forward(self):
        N, T, D, H = 2, 3, 4, 5
        rnn = RNNLayer(4,5)
        x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
        prev_h = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
        rnn.Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
        rnn.Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
        rnn.b = np.linspace(-0.7, 0.1, num=H)

        h = rnn.forward(x, prev_h)
        expected_h = np.array([
          [
            [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
            [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
            [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
          ],
          [
            [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
            [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
            [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
            
        assert rel_error(expected_h[0], h[0]) < 1e-6

    def test_backward_step(self):
        np.random.seed(145)
        N, D, H = 3, 10, 5
        rnn = RNNLayer(D, H)

        x = np.random.randn(N, D)
        prev_h = np.random.randn(N, H)
        rnn.Wx = np.random.randn(D, H)
        rnn.Wh = np.random.randn(H, H)
        rnn.b = np.random.randn(H)

        out, cache = rnn.forward_step(x, prev_h)

        dnext_h = np.linspace(-0.2, 0.4, num=N*H).reshape(N, H)

        dx, dprev_h, dWx, dWh, db = rnn.backward_step(dnext_h, cache)
        f = lambda _: rnn.forward_step(x, prev_h)[0]


        dx_num = grad_check(f, x, dnext_h)
        dprev_h_num = grad_check(f, prev_h, dnext_h)
        dWx_num = grad_check(f, rnn.Wx, dnext_h)
        dWh_num = grad_check(f, rnn.Wh, dnext_h)
        db_num = grad_check(f, rnn.b, dnext_h)
        
        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dprev_h_num, dprev_h) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6
        
    def test_backward(self):
        np.random.seed(145)

        N, D, T, H = 3, 10, 7, 5
        rnn = RNNLayer(D, H)

        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        rnn.Wx = np.random.randn(D, H)
        rnn.Wh = np.random.randn(H, H)
        rnn.b = np.random.randn(H)

        out = rnn.forward(x, h0)

        dnext_h = np.random.randn(*out.shape)

        rnn.backward(dnext_h)

        dx, dh0, dWx, dWh, db = rnn.grad['dx'], rnn.grad['dh0'], rnn.grad['dWx'], rnn.grad['dWh'], rnn.grad['db']

        f = lambda _: rnn.forward(x, h0)

        dx_num = grad_check(f, x, dnext_h)
        dh0_num = grad_check(f, h0, dnext_h)
        dWx_num = grad_check(f, rnn.Wx, dnext_h)
        dWh_num = grad_check(f, rnn.Wh, dnext_h)
        db_num = grad_check(f, rnn.b, dnext_h)

        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dh0_num, dh0) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6

if __name__ == '__main__':
    unittest.main()

