import unittest
import numpy as np

from blg561.layer.recurrent_layers import LSTMLayer
from blg561.checks import rel_error, grad_check

class TestLSTM(unittest.TestCase):

    def test_forward_step(self):
        N, D, H = 3, 4, 5
        lstm = LSTMLayer(4, 5)
        x = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)
        prev_c = np.linspace(-0.4, 0.9, num=N*H).reshape(N, H)
        lstm.Wx = np.linspace(-2.1, 1.3, num=4*D*H).reshape(D, 4 * H)
        lstm.Wh = np.linspace(-0.7, 2.2, num=4*H*H).reshape(H, 4 * H)
        lstm.b = np.linspace(0.3, 0.7, num=4*H)

        next_h, next_c, _ = lstm.forward_step(x, prev_h, prev_c)

        expected_next_h = np.asarray([
            [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],
            [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],
            [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]])
        expected_next_c = np.asarray([
            [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],
            [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],
            [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]])

        assert rel_error(expected_next_h, next_h) < 1e-6
        assert rel_error(expected_next_c, next_c) < 1e-6

    def test_forward(self):
        N, D, H, T = 2, 5, 4, 3
        lstm = LSTMLayer(5, 4)
        x = np.linspace(-0.4, 0.6, num=N*T*D).reshape(N, T, D)
        h0 = np.linspace(-0.4, 0.8, num=N*H).reshape(N, H)
        lstm.Wx = np.linspace(-0.2, 0.9, num=4*D*H).reshape(D, 4 * H)
        lstm.Wh = np.linspace(-0.3, 0.6, num=4*H*H).reshape(H, 4 * H)
        lstm.b = np.linspace(0.2, 0.7, num=4*H)

        h = lstm.forward(x, h0)

        expected_h = np.asarray([
         [[ 0.01764008,  0.01823233,  0.01882671,  0.0194232 ],
          [ 0.11287491,  0.12146228,  0.13018446,  0.13902939],
          [ 0.31358768,  0.33338627,  0.35304453,  0.37250975]],
         [[ 0.45767879,  0.4761092,   0.4936887,   0.51041945],
          [ 0.6704845,   0.69350089,  0.71486014,  0.7346449 ],
          [ 0.81733511,  0.83677871,  0.85403753,  0.86935314]]])
            
        assert rel_error(expected_h[0], h[0]) < 1e-6

    def test_backward_step(self):
        np.random.seed(132)

        N, D, H = 4, 5, 6
        lstm = LSTMLayer(5, 6)
        x = np.random.randn(N, D)
        prev_h = np.random.randn(N, H)
        prev_c = np.random.randn(N, H)
        lstm.Wx = np.random.randn(D, 4 * H)
        lstm.Wh = np.random.randn(H, 4 * H)
        lstm.b = np.random.randn(4 * H)

        next_h, next_c, cache = lstm.forward_step(x, prev_h, prev_c)

        dnext_h = np.random.randn(*next_h.shape)
        dnext_c = np.random.randn(*next_c.shape)

        f_h = lambda _: lstm.forward_step(x, prev_h, prev_c)[0]
        f_c = lambda _: lstm.forward_step(x, prev_h, prev_c)[1]

        dx_num = grad_check(f_h, x, dnext_h) + grad_check(f_c, x, dnext_c)
        dprev_h_num = grad_check(f_h, prev_h, dnext_h) + grad_check(f_c, prev_h, dnext_c)
        dprev_c_num = grad_check(f_h, prev_c, dnext_h) + grad_check(f_c, prev_c, dnext_c)
        dWx_num = grad_check(f_h, lstm.Wx, dnext_h) + grad_check(f_c, lstm.Wx, dnext_c)
        dWh_num = grad_check(f_h, lstm.Wh, dnext_h) + grad_check(f_c, lstm.Wh, dnext_c)
        db_num = grad_check(f_h, lstm.b, dnext_h) + grad_check(f_c, lstm.b, dnext_c)

        dx, dh, dc, dWx, dWh, db = lstm.backward_step(dnext_h, dnext_c, cache)
        
        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dprev_h_num, dh) < 1e-6
        assert rel_error(dprev_c_num, dc) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6
        
    def test_backward(self):
        np.random.seed(231)

        N, D, T, H = 2, 3, 10, 6

        lstm = LSTMLayer(3, 6)

        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        lstm.Wx = np.random.randn(D, 4 * H)
        lstm.Wh = np.random.randn(H, 4 * H)
        lstm.b = np.random.randn(4 * H)

        out = lstm.forward(x, h0)

        dnext_h = np.random.randn(*out.shape)

        lstm.backward(dnext_h)
        dx, dh0, dWx, dWh, db = lstm.grad['dx'], lstm.grad['dh0'], lstm.grad['dWx'], lstm.grad['dWh'], lstm.grad['db']

        f = lambda _: lstm.forward(x, h0)

        dx_num = grad_check(f, x, dnext_h)
        dh0_num = grad_check(f, h0, dnext_h)
        dWx_num = grad_check(f, lstm.Wx, dnext_h)
        dWh_num = grad_check(f, lstm.Wh, dnext_h)
        db_num = grad_check(f, lstm.b, dnext_h)

        assert rel_error(dx_num, dx) < 1e-6
        assert rel_error(dh0_num, dh0) < 1e-6
        assert rel_error(dWx_num, dWx) < 1e-6
        assert rel_error(dWh_num, dWh) < 1e-6
        assert rel_error(db_num, db) < 1e-6

if __name__ == '__main__':
    unittest.main()
