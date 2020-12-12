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
    time.sleep(0.1)    # Wait until shapes appear

    game_size = get_monitor_resolution()
    game_region = (0, 0, game_size[0], game_size[1])

    # Check if the button is within the game region
    if not all_shapes_button[0] < game_size[0] and all_shapes_button[1] < game_size[1]:
        raise GameSizeNotCorrect("Please set the game_size variable manually.")

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

    back_button = center_of_button_on_screen("assets/back-button.png", "back")
    pyautogui.click(back_button)
    pyautogui.press("esc")
