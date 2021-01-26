# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: part1
# @ Date: 26-Jan-2021
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2021 Batuhan Faik Derinbay
# @ Project: VisionProject
# @ Description: Play the Dice Game
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import time
import cv2
import numpy as np
import pyautogui

import common.methods as game
from common.exceptions import GameNotInitiatedCorrectly


def auto_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    # return the edged image
    return edged


def number_of_circles(img):
    circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=35,
                              minRadius=0, maxRadius=0)
    try:
        no_circles = len(circle[0])
    except TypeError:
        no_circles = 0
    return no_circles


def play_part1(ss):
    edged = auto_canny(ss)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cropped_die = []
    for contour in contours:
        die_top_left = contour.min(axis=0)[0]
        die_bot_right = contour.max(axis=0)[0]
        cropped_die.append(
            edged[die_top_left[1]:die_bot_right[1], die_top_left[0]:die_bot_right[0]])

    return number_of_circles(cropped_die[2]), number_of_circles(cropped_die[1]), number_of_circles(cropped_die[0])


if __name__ == "__main__":
    # Ask the user game they would like to play
    print("Waiting for the user to choose the game...")
    part = pyautogui.confirm(text="Which game would you like to play?", title="Game Selection",
                             buttons=['Part 1', 'Part 2'])
    rounds = int(pyautogui.confirm(text="For how many dice rolls would you like to play?",
                                   title="Number of Rounds",
                                   buttons=[10, 50, 69, 100, 250, 420, 500]))
    print("Now playing {} for {} rounds.".format(part, rounds))

    # Switch to the game within 5 seconds (Tested only on Firefox)
    mode = game.prepare_web_game(5)

    if part == "Part 1" or part == "Part 2":  # Play the song
        game_region = game.go_to_page(part, mode)
    else:  # No song selected
        raise GameNotInitiatedCorrectly("No game has been selected")

    region_left, region_top = game_region[0], game_region[1]  # w, h
    region_w, region_h = int(game_region[2]), int(game_region[3] * 0.5)
    control_region = (region_left, region_top, region_w, region_h)  # Left, Top, W, H

    for i in range(rounds):
        ss = np.array(pyautogui.screenshot(region=control_region))
        ss = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)
        if part == "Part 1":
            die_A, die_S, die_D = play_part1(ss)
        else:
            die_A, die_S, die_D = play_part1(ss)

        if die_A > die_S and die_A > die_D:
            pyautogui.press("a")
        elif die_S > die_D:
            pyautogui.press("s")
        else:
            pyautogui.press("d")

        # Random sleep is used to make sure that dice are different every round in part 2
        random_sleep = np.random.uniform(0.2, 0.4)
        time.sleep(random_sleep)  # Wait until new dice appear

    print("{} rounds are over. Check the accuracy.".format(rounds))
    time.sleep(5)  # Wait until the dance is complete
    game.back_to_original_state()
    print("Game completed.")
