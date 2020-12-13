# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: part4
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description Play the dancing game
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import time

import cv2
import numpy as np
import pyautogui

import common.methods as game
from common.detector import corner_detector
from common.exceptions import GameNotInitiatedCorrectly


def check_if_all_pixels_are_black(img, threshold):
    return True if np.sum(img) < threshold else False


if __name__ == "__main__":
    # Switch to the game within 10 seconds (Tested only on Firefox)
    game.prepare_web_game(10)

    song = 2
    if song == 1:  # Play "Vabank"
        vabank_button, button_name = "assets/vabank-button.png", "vabank"
        game_region = game.go_to_page(vabank_button, button_name)
    elif song == 2:  # Play "Shame"
        shame_button, button_name = "assets/shame-button.png", "shame"
        game_region = game.go_to_page(shame_button, button_name)
    else:  # No song selected
        print("No song has been selected")
        raise GameNotInitiatedCorrectly()

    region_left, region_top = game_region[2] // 2, int(game_region[3] * 0.76)  # w, h
    region_w, region_h = int(game_region[2] * 0.115), int(game_region[3] - region_top)
    control_region = (region_left + region_w // 2.4, region_top, int(region_w + region_w // 2), region_h)

    patch_w, patch_h = int(region_w // 2), int(region_h // 2)

    no_of_presses = 0
    while no_of_presses < 18:
        # Take a screenshot around the control region
        ss = np.array(pyautogui.screenshot(region=control_region))
        if check_if_all_pixels_are_black(ss[patch_h:patch_h+3, patch_w:patch_w+3, :], 100):
            ss_gray = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)
            # Calculate the edges
            edges = cv2.Canny(ss_gray, 200, 200, apertureSize=5, L2gradient=True)
            corners, shapes_marked = corner_detector(edges, 3, 15000, 0.05, 3, "harris")
            if len(corners) > 40:
                pyautogui.press("d")
            elif len(corners) > 20:
                pyautogui.press("s")
            elif len(corners) > 16:
                pyautogui.press("f")
            else:
                pyautogui.press("a")

            no_of_presses += 1

    time.sleep(5)    # Wait until the dance is complete
    game.back_to_original_state()
