# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: q5i
# @ Date: 11-Mar-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: blg354e_assignment1
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np


def a_k(k):
    return (-1 / (2 * np.pi * k)) * (np.sin(np.pi * k / 2) + np.exp(-1j * np.pi / 2 * (k + 1)) * (np.exp(-1j * np.pi * k) - 1))

k_min = -100
k_max = 100
a_0 = 0.75
T_0 = 20

f_0 = 1 / T_0
t = np.arange(-25, 25, 0.01)  # start,stop,step
x_t = a_0

for k in range(k_min, k_max + 1):
    if not k == 0:
        x_t += a_k(k) * np.exp(1j * 2 * np.pi * k * f_0 * t)

plt.plot(t, x_t)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig("q5i_v.png")
