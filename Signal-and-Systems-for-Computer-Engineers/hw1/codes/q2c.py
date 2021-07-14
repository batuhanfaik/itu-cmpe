# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: q2b
# @ Date: 10-Mar-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: blg354e_assignment1
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np

w = 1
t = np.arange(-2 * np.pi, 2 * np.pi, 0.1)  # start,stop,step
x_t = 5.21*np.cos(w*t - 2.829 * np.pi)
plt.plot(t, x_t)
plt.grid()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend(['cos(wt-2.829pi)'])
plt.savefig("q2c.png")
