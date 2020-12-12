# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: part3
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description: Implementation of minimum eigenvalue corner detector
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import cv2
import numpy as np
import pyautogui

import common.methods as game


def harris_corner_detector(img, block_size=3, threshold=10000, k=0.05, downscaling_factor=1):
    corners = []
    output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

    # Downscale the input image for faster processing
    if downscaling_factor > 1:
        w = int(img.shape[1] * 1/downscaling_factor)
        h = int(img.shape[0] * 1/downscaling_factor)
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
            r = np.linalg.det(g) - k * np.square(np.trace(g))
            if r > threshold:
                x_, y_ = int(x * upscale_ratio), int(y * upscale_ratio)
                p = int(np.ceil(upscale_ratio / 2))    # padding
                corners.append([x_, y_, r])
                output_img[y_-p:y_+p, x_-p:x_+p] = (0, 255, 0)

    return corners, output_img


if __name__ == "__main__":
    # Switch to the game within 10 seconds (Tested only on Firefox)
    game.prepare_web_game(10)

    # Go to all shapes screen
    all_shapes_button, button_name = "assets/all-shapes-button.png", "all shapes"
    game_region = game.go_to_page(all_shapes_button, button_name)
    # Take a screenshot of the game
    ss = np.array(pyautogui.screenshot(region=game_region))
    ss_gray = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)
    game.back_to_original_state()

    # Calculate the edges
    hysteresis_thresholds = (200, 200)
    aperture_size = 5
    l2_gradient = True
    edges = cv2.Canny(ss_gray, *hysteresis_thresholds, apertureSize=aperture_size,
                      L2gradient=l2_gradient)

    # Crop the edges
    y_upper_bound = int(0.38 * game_region[3])
    y_lower_bound = int(0.62 * game_region[3])
    shapes = edges[y_upper_bound:y_lower_bound, :]

    # The values for the corner detector are as follows:
    #    Input image = cropped shapes
    #    Block size = 3x3
    #    Threshold = 150000
    #    k = 0.05
    #    Downscaling factor = 3
    corners, shapes_marked = harris_corner_detector(shapes, 3, 15000, 0.05, 3)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_colored[y_upper_bound:y_lower_bound, :] = shapes_marked

    cv2.imwrite("output_corners.png", edges_colored)
