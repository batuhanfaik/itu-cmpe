import unittest
import numpy as np

from blg561.layer.recurrent_layers import GRULayer
from blg561.checks import rel_error, grad_check

class TestGRU(unittest.TestCase):

    def test_forward_step(self):
        N, D, H = 3, 4, 5
        gru = GRULayer(4, 5)
        x = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)
        gru.Wx = np.linspace(-2.1, 1.3, num=2*D*H).reshape(D, 2 * H)
        gru.Wh = np.linspace(-0.7, 2.2, num=2*H*H).reshape(H, 2 * H)
        gru.b = np.linspace(0.3, 0.7, num=2*H)
        gru.Wxi = np.linspace(-1.8, 2.1, num=D*H).reshape(D, H)
        gru.Whi = np.linspace(-0.9, 1.6, num=H*H).reshape(H, H)
        gru.bi = np.linspace(0.1, 0.9, num=H)

        next_h, _ = gru.forward_step(x, prev_h)

        expected_next_h = np.asarray([
            [-0.0999449,  -0.03125071,  0.03639522,  0.10290262,  0.16817868],
            [ 0.2976273,   0.36884449,  0.3992976,   0.42559245,  0.45798687],
            [ 0.49371886,  0.70402669,  0.70335355,  0.71138395,  0.7425779 ]])

        assert rel_error(expected_next_h, next_h) < 1e-6

    def test_forward(self):
        N, D, H, T = 2, 5, 4, 3
        gru = GRULayer(5, 4)
        x = np.linspace(-0.4, 0.6, num=N*T*D).reshape(N, T, D)
        h0 = np.linspace(-0.4, 0.8, num=N*H).reshape(N, H)
        gru.Wx = np.linspace(-0.2, 0.9, num=2*D*H).reshape(D, 2 * H)
        gru.Wh = np.linspace(-0.3, 0.6, num=2*H*H).reshape(H, 2 * H)
        gru.b = np.linspace(0.2, 0.7, num=2*H)
        gru.Wxi = np.linspace(-0.4, 1.6, num=D*H).reshape(D, H)
        gru.Whi = np.linspace(-0.7, 0.4, num=H*H).reshape(H, H)
        gru.bi = np.linspace(0.4, 0.9, num=H)

        h = gru.forward(x, h0)

        expected_h = np.asarray([
         [[-0.19332136, -0.12098466, -0.0478229,   0.02618775],
          [ 0.03427635,  0.09687263,  0.15884181,  0.22017801],
          [ 0.22173995,  0.30237624,  0.36888384,  0.42432841]],

         [[ 0.3842619,   0.55153297,  0.69594892,  0.83361145],
          [ 0.48151744,  0.63219579,  0.74877489,  0.85939455],
          [ 0.56501369,  0.69358245,  0.78641933,  0.87723583]]])
            
        assert rel_error(expected_h[0], h[0]) < 1e-6

    def test_backward_step(self):
        np.random.seed(145)

        N, D, H = 4, 5, 6
        gru = GRULayer(5, 6)
        x = np.random.randn(N, D)
        prev_h = np.random.randn(N, H)
        gru.Wx = np.random.randn(D, 2 * H)
        gru.Wh = np.random.randn(H, 2 * H)
        gru.b = np.random.randn(2 * H)
        gru.Wxi = np.random.randn(D, H)
        gru.Whi = np.random.randn(H, H)
        gru.bi = np.random.randn(H)

        next_h, cache = gru.forward_step(x, prev_h)

        dnext_h = np.random.randn(*next_h.shape)

        f = lambda _: gru.forward_step(x, prev_h)[0]


        dx_num = grad_check(f, x, dnext_h)
        dprev_h_num = grad_check(f, prev_h, dnext_h)
        dWx_num = grad_check(f, gru.Wx, dnext_h)
        dWh_num = grad_check(f, gru.Wh, dnext_h)
        db_num = grad_check(f, gru.b, dnext_h)
        dWxi_num = grad_check(f, gru.Wxi, dnext_h)
        dWhi_num = grad_check(f, gru.Whi, dnext_h)
        dbi_num = grad_check(f, gru.bi, dnext_h)

        dx, dprev_h, dWx, dWh, db, dWxi, dWhi, dbi = gru.backward_step(dnext_h, cache)
        
        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dprev_h_num, dprev_h) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6
        assert rel_error(dWxi_num, dWxi) < 1e-6
        assert rel_error(dWhi_num, dWhi) < 1e-6
        assert rel_error(dbi_num, dbi) < 1e-6
        
    def test_backward(self):
        np.random.seed(145)

        N, D, T, H = 2, 3, 10, 6
        gru = GRULayer(3, 6)
        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        gru.Wx = np.random.randn(D, 2 * H)
        gru.Wh = np.random.randn(H, 2 * H)
        gru.b = np.random.randn(2 * H)
        gru.Wxi = np.random.randn(D, H)
        gru.Whi = np.random.randn(H, H)
        gru.bi = np.random.randn(H)

        out = gru.forward(x, h0)

        dnext_h = np.random.randn(*out.shape)

        gru.backward(dnext_h)

        dx, dh0, dWx, dWh, db, dWxi, dWhi, dbi = gru.grad['dx'], gru.grad['dh0'], gru.grad['dWx'], \
                                                 gru.grad['dWh'], gru.grad['db'], \
                                                 gru.grad['dWxi'], gru.grad['dWhi'], gru.grad['dbi']

        f = lambda _: gru.forward(x, h0)

        dx_num = grad_check(f, x, dnext_h)
        dh0_num = grad_check(f, h0, dnext_h)
        dWx_num = grad_check(f, gru.Wx, dnext_h)
        dWh_num = grad_check(f, gru.Wh, dnext_h)
        db_num = grad_check(f, gru.b, dnext_h)
        dWxi_num = grad_check(f, gru.Wxi, dnext_h)
        dWhi_num = grad_check(f, gru.Whi, dnext_h)
        dbi_num = grad_check(f, gru.bi, dnext_h)

        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dh0_num, dh0) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6
        assert rel_error(dWxi_num, dWxi) < 1e-6
        assert rel_error(dWhi_num, dWhi) < 1e-6
        assert rel_error(dbi_num, dbi) < 1e-6

if __name__ == '__main__':
    unittest.main()
