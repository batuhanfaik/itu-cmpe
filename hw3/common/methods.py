# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: methods
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: hw3
# @ Description: Useful static methods
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pyautogui
import time

from common.exceptions import GameNotInitiatedCorrectly, GameSizeNotCorrect


def get_resolution():
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


def prepare_web_game(secs=10):
    if not pyautogui.locateOnScreen("assets/fullscreen-button.png"):
        pyautogui.alert(text="Please launch the game within the next {} seconds.".format(secs),
                        title="Game is not ready!", button='OK')
        time.sleep(secs)
        if not pyautogui.locateOnScreen("assets/fullscreen-button.png"):
            raise GameNotInitiatedCorrectly()


def go_to_page(button_img_path, button_name=""):
    fullscreen_button = center_of_button_on_screen("assets/fullscreen-button.png", "fullscreen")
    pyautogui.click(fullscreen_button)
    time.sleep(4)    # Wait until fullscreen notification is closed
    button = center_of_button_on_screen(button_img_path, button_name)
    pyautogui.click(button)
    time.sleep(0.25)    # Wait until shapes appear
    game_size = get_resolution()
    game_region = (0, 0, game_size[0], game_size[1])

    # Check if the button is within the game region
    if not button[0] < game_size[0] and button[1] < game_size[1]:
        raise GameSizeNotCorrect("Please set the game_size variable manually.")

    return game_region


def back_to_original_state():
    try:
        back_button = center_of_button_on_screen("assets/back-button.png", "back")
        pyautogui.click(back_button)
        pyautogui.press("esc")
    except GameNotInitiatedCorrectly:
        pyautogui.press("esc")
        pyautogui.press("esc")
