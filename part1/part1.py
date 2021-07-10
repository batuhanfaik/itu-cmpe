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


def die_locations(i):
    return i[1]


def warp_die(img, rect):
    tl, tr, br, bl = rect

    width_A = np.sqrt(np.power((br[0] - bl[0]), 2) + np.power((br[1] - bl[1]), 2))
    width_B = np.sqrt(np.power((tr[0] - tl[0]), 2) + np.power((tr[1] - tl[1]), 2))
    max_w = max(int(width_A), int(width_B))

    height_A = np.sqrt(np.power((tr[0] - br[0]), 2) + np.power((tr[1] - br[1]), 2))
    height_B = np.sqrt(np.power((tl[0] - bl[0]), 2) + np.power((tl[1] - bl[1]), 2))
    max_h = max(int(height_A), int(height_B))

    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    return warped


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


def play_part2(ss):
    _, thresh = cv2.threshold(ss, 170, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    die_faces = []
    index = 0

    while len(die_faces) < 3 and index < len(contours):
        c = contours[index]
        c_area = cv2.contourArea(c) > 50000
        if c_area:
            die_faces.append(c)
        index += 1

    numbers_on_faces = []
    for face_contour in die_faces:
        face_cntr = face_contour[:, 0, :]
        top_left = face_cntr[face_cntr.sum(axis=1).argmin()]
        top_right = face_cntr[(face_cntr.T[0] - face_cntr.T[1]).argmax()]
        bot_right = face_cntr[face_cntr.sum(axis=1).argmax()]
        bot_left = face_cntr[(face_cntr.T[0] - face_cntr.T[1]).argmin()]
        points = np.vstack((top_left, top_right, bot_right, bot_left)).astype(np.float32)
        warped_die = warp_die(thresh, points)
        no_points = len(
            cv2.HoughCircles(warped_die, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=20,
                             minRadius=0, maxRadius=0)[0])
        numbers_on_faces.append((no_points, top_left[0]))

    return sorted(numbers_on_faces, key=die_locations)


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
    region_w, region_h = int(game_region[2]), int(game_region[3] * 0.56)
    control_region = (region_left, region_top, region_w, region_h)  # Left, Top, W, H

    for i in range(rounds):
        ss = np.array(pyautogui.screenshot("screenshots/ss/img_{}.png".format(i), region=control_region))
        ss = cv2.cvtColor(ss, cv2.COLOR_RGB2GRAY)
        if part == "Part 1":
            die_A, die_S, die_D = play_part1(ss)
        else:
            die_A, die_S, die_D = play_part2(ss)

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
