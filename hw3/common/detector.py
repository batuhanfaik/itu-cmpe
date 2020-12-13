# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: detector
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description: Corner detector implementation
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import numpy as np


def corner_detector(img, block_size=3, threshold=10000, k=0.05, downscaling_factor=1,
                    detector="harris"):
    corners = []
    output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

    # Downscale the input image for faster processing
    if downscaling_factor > 1:
        w = int(img.shape[1] * 1 / downscaling_factor)
        h = int(img.shape[0] * 1 / downscaling_factor)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        upscale_ratio = downscaling_factor
    else:
        upscale_ratio = 1

    # Smooth out the image
    smoothing_kernel = np.ones((block_size, block_size), dtype=np.float32) / np.square(block_size)
    img = cv2.filter2D(img, -1, smoothing_kernel)

    offset = block_size // 2
    y_range = img.shape[0] - offset
    x_range = img.shape[1] - offset

    dy, dx = np.gradient(img)
    Ixx = dx * dx
    Ixy = dy * dx
    Iyy = dy * dy

    x_prev, y_prev = -255, -255

    for y in range(offset, y_range):
        for x in range(offset, x_range):
            # Coordinates of the window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            # Ixx, Ixy, Iyy per window
            w_Ixx = np.sum(Ixx[start_y: end_y, start_x: end_x])
            w_Ixy = np.sum(Ixy[start_y: end_y, start_x: end_x])
            w_Iyy = np.sum(Iyy[start_y: end_y, start_x: end_x])

            # G matrix
            g = np.array([[w_Ixx, w_Ixy], [w_Ixy, w_Iyy]])

            # Calculate R = C(G)
            if detector == "shi":
                r = min(g[0][0], g[1][1])
            else:
                r = np.linalg.det(g) - k * np.square(np.trace(g))
            if r > threshold:
                x_, y_ = int(x * upscale_ratio), int(y * upscale_ratio)
                p = int(np.ceil(upscale_ratio / 2))  # padding
                output_img[y_-p:y_+p, x_-p:x_+p] = (0, 255, 0)
                if not (x_prev-8 < x < x_prev+8 and y_prev-8 < y < y_prev+8):
                    corners.append([x_, y_, r])
                    x_prev, y_prev = x, y

    return corners, output_img
