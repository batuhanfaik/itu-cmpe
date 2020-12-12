# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: part1
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description: Takes a screenshot of the All shapes, applies Sobel filter and saves
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import cv2
import numpy as np
import pyautogui
import time

import common.methods as game

if __name__ == "__main__":
    # Switch to the game (Tested only on Firefox)
    time.sleep(10)

    # Go to all shapes screen
    all_shapes_button, button_name = "assets/all-shapes-button.png", "all shapes"
    game_region = game.go_to_page(all_shapes_button, button_name)
    ss = np.array(pyautogui.screenshot(region=game_region))
    ss_gray = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)

    # Define filters (sizes available are 3 and 5)
    filter_size = 3
    if filter_size == 5:
        sobel_kernel_y = np.array([[1, 2, 3, 2, 1],
                                   [2, 3, 5, 3, 2],
                                   [0, 0, 0, 0, 0],
                                   [-2, -3, -5, -3, -2],
                                   [-1, -2, -3, -2, -1]])
        sobel_kernel_x = np.transpose(sobel_kernel_y)
    else:
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_x = np.transpose(sobel_kernel_y)

    sobel_x = cv2.filter2D(ss_gray, -1, sobel_kernel_x)
    sobel_y = cv2.filter2D(ss_gray, -1, sobel_kernel_y)
    edge_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edge_magnitude *= 255 / np.max(edge_magnitude)

    cv2.imwrite("output.png", edge_magnitude.astype(np.uint8))

    game.back_to_original_state()
