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
from common.detector import corner_detector


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

    # Optimal values for the corner detector for fast processing are as follows
    # With the current values, the processing will take some time
    #    Input image = cropped shapes
    #    Block size = 3x3
    #    Threshold = 150000
    #    k = 0.05
    #    Downscaling factor = 3
    #    Detector = "harris"
    corners, shapes_marked = corner_detector(shapes, 9, 15000, 0.05, 1, "harris")
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_colored[y_upper_bound:y_lower_bound, :] = shapes_marked

    cv2.imwrite("output_corners.png", edges_colored)
