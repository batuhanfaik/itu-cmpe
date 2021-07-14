# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: q5ii
# @ Date: 11-Mar-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: blg354e_assignment1
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np


def a_k(k):
    return (4j * np.pi * k - 3 * np.exp(1j * 4/3 * np.pi * k) + 3)/(2 * np.pi ** 2 * k ** 2) - (6j * np.pi * k + 9 * np.exp(-1j * 2/3 * np.pi * k) - 9)/(3 * np.pi ** 2 * k ** 2)


k_min = 1
k_max = 100
a_0 = 1
T_0 = 3

f_0 = 1 / T_0
t = np.arange(-5, 5, 0.01)  # start,stop,step
x_t = a_0

for k in range(k_min, k_max + 1):
    if not k == 0:
        x_t += a_k(k) * np.exp(1j * 2 * np.pi * k * f_0 * t)

plt.plot(t, x_t)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig("q5ii.png")
