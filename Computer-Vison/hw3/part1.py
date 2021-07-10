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

import common.methods as game

if __name__ == "__main__":
    # Switch to the game within 5 seconds (Tested only on Firefox)
    mode = game.prepare_web_game(5)

    # Go to all shapes screen
    game_region = game.go_to_page("All-Shapes", mode)
    # Take a screenshot of the game
    ss = np.array(pyautogui.screenshot(region=game_region))
    ss_gray = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)

    game.back_to_original_state()

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

    cv2.imwrite("output_sobel.png", edge_magnitude.astype(np.uint8))

    print("Part 1 completed. Check the outputs.")
