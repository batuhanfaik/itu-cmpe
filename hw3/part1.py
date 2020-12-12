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

from common.exceptions import GameNotInitiatedCorrectly, GameSizeNotCorrect


def get_monitor_resolution():
    w, h = pyautogui.size()
    # If the width is more than twice the height there probably are multiple monitors
    # For more than two monitors, the code is likely to fail
    if w/h > 2/1:
        w //= 2
    return w, h


def center_of_button_on_screen(button_img_path, button_name=""):
    button = pyautogui.locateOnScreen(button_img_path)
    if not button:
        raise GameNotInitiatedCorrectly("Couldn't locate the {} button!".format(button_name))
    else:
        return pyautogui.center(button)


if __name__ == "__main__":
    # Switch to the game (Tested only on Firefox)
    # time.sleep(5)

    fullscreen_button = center_of_button_on_screen("assets/fullscreen-button.png", "fullscreen")
    pyautogui.click(fullscreen_button)
    time.sleep(4)    # Wait until fullscreen notification is closed
    all_shapes_button = center_of_button_on_screen("assets/all-shapes-button.png", "all shapes")
    pyautogui.click(all_shapes_button)

    game_size = get_monitor_resolution()
    game_region = (0, 0, game_size[0], game_size[1])

    # Check if the button is within the game region
    if not all_shapes_button[0] < game_size[0] and all_shapes_button[1] < game_size[1]:
        raise GameSizeNotCorrect("Please set the game_size variable manually.")

    ss = pyautogui.screenshot(region=game_region)
    ss_cv = np.array(ss)
    ss_gray = cv2.cvtColor(ss_cv, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(ss_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(ss_gray, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imwrite("sobel_x.png", sobel_x)
    cv2.imwrite("sobel_y.png", sobel_y)

    back_button = center_of_button_on_screen("assets/back-button.png", "back")
    pyautogui.click(back_button)
    pyautogui.press("esc")
