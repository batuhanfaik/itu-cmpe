# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: q5iii
# @ Date: 11-Mar-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: blg354e_assignment1
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np


def a_k(k):
    return 0.25 * ((0.32 * (1 + np.exp(-1j * np.pi * k)))/(k ** 2 - 1) -
                   ((4j * (1 - np.exp(-1j * np.pi * k)))/(np.pi * k)))


k_min = 0
k_max = 100
a_0 = 0.84
T_0 = 4

f_0 = 1 / T_0
t = np.arange(-6, 6, 0.01)  # start,stop,step
x_t = a_0

for k in range(k_min, k_max + 1):
    if not k == 0 and not k == 1:
        x_t += a_k(k) * np.exp(1j * 2 * np.pi * k * f_0 * t)

plt.plot(t, x_t)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig("q5iii.png")
