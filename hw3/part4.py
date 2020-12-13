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
from scipy.ndimage import label

import common.methods as game
from common.exceptions import GameNotInitiatedCorrectly


def check_if_all_pixels_are_black(img, threshold):
    return True if np.sum(img) < threshold else False


if __name__ == "__main__":
    # Ask the user which song they would like to play
    print("Waiting for the user to choose a song...")
    song = pyautogui.confirm(text="Which song would you like to play?", title="Song Selection",
                             buttons=['Vabank', 'Shame'])
    print("Now playing: {}".format(song))

    # Switch to the game within 10 seconds (Tested only on Firefox)
    game.prepare_web_game(10)

    if song == "Vabank":  # Play "Vabank"
        vabank_button, button_name = "assets/vabank-button.png", song
        game_region = game.go_to_page(vabank_button, button_name)
    elif song == "Shame":  # Play "Shame"
        shame_button, button_name = "assets/shame-button.png", song
        game_region = game.go_to_page(shame_button, button_name)
    else:  # No song selected
        raise GameNotInitiatedCorrectly("No song has been selected")

    # Capturing region calculations
    region_left, region_top = game_region[2] // 2, int(game_region[3] * 0.76)  # w, h
    region_w, region_h = int(game_region[2] * 0.115), int(game_region[3] - region_top)
    control_region = (region_left + region_w // 2.4, region_top, int(region_w + region_w // 2), region_h)

    patch_w, patch_h = int(region_w // 2), int(region_h // 2)

    no_of_presses = 0
    while no_of_presses < 18:
        # Take a screenshot around the control region
        ss = np.array(pyautogui.screenshot(region=control_region))
        # Patch that checks if all pixels are black is 3x3
        if check_if_all_pixels_are_black(ss[patch_h:patch_h+3, patch_w:patch_w+3, :], 10):
            ss_gray = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)
            # Get a 2D matrix containing r values for corners
            dst = cv2.dilate(cv2.cornerHarris(ss_gray, 5, 3, 0.04), None)
            # Mask corner regions
            regions = np.zeros(ss_gray.shape)
            regions[dst > 0.01 * np.max(dst)] = 1
            # Count the number of corners
            _, num_corners = label(regions)

            time.sleep(0.2)    # Wait until the object is within the recognition area
            # Press the necessary keys
            if num_corners == 10:    # Star
                pyautogui.press("d")
            elif num_corners == 6:    # Hexagon
                pyautogui.press("f")
            elif num_corners == 4:    # Square
                pyautogui.press("s")
            else:    # Triangle
                pyautogui.press("a")

            no_of_presses += 1
            time.sleep(0.4)    # Wait until the object disappears

    print("Song finished. Check the accuracy.")
    time.sleep(5)    # Wait until the dance is complete
    game.back_to_original_state()
    print("Part 4 completed.")
