# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: part2
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description: Takes a screenshot of the All shapes, applies Canny filter and saves
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

    # Calculate the edges
    hysteresis_thresholds = (200, 200)
    aperture_size = 5
    l2_gradient = True
    edges = cv2.Canny(ss_gray, *hysteresis_thresholds, apertureSize=aperture_size,
                      L2gradient=l2_gradient)

    cv2.imwrite("output_canny.png", edges)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Draw contours
    cv2.drawContours(ss, contours, -1, (0, 255, 0), 3)
    cv2.imwrite("output_contours.png", ss)
    print("Part 2 completed. Check the outputs.")
