# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: q5iv
# @ Date: 10-Mar-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: blg354e_assignment1
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np


def a_k(k):
    return 0.5 * ((1 - np.exp(-1j * np.pi * k - 1)) / (1j * np.pi * k + 1))
    # return 0.5 * (1 - np.exp(-1) * (-1) ** k) / (np.pi * 1j * k + 1)


k_min = 0
k_max = 10
a_0 = 0.35
T_0 = 2

f_0 = 1 / T_0
t = np.arange(-5, 5, 0.01)  # start,stop,step
x_t = a_0
# print(x_t)

for k in range(k_min, k_max + 1):
    if not k == 0:
        x_t += a_k(k) * np.exp(1j * 2 * np.pi * k * f_0 * t)
    # print(x_t)

plt.plot(t, x_t)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig("q5iv.png")
