# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: exceptions
# @ Date: 12-Dec-2020
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2020 Batuhan Faik Derinbay
# @ Project: VisionProject
# @ Description: exceptions defined within the project
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from pyscreeze import ImageNotFoundException


class GameNotInitiatedCorrectly(ImageNotFoundException):
    def __init__(self, message=""):
        self.message = "Game is not started correctly or not fully visible.\n" \
                       "Please launch it on Firefox, make sure nothing obstructs the game on the " \
                       "primary monitor and don't enter fullscreen" \
                       "mode. The program will do so automatically.\n" \
                       "The game is hosted on the following link: " \
                       "https://web.itu.edu.tr/sahinyu/cube_project_web/"
        self.message += message
        super().__init__(self.message)


class GameSizeNotCorrect(Exception):
    def __init__(self, message=""):
        self.message = "Game size is not correct. There probably was an error during calculation " \
                       "of the screen size.\n"
        self.message += message
        super().__init__(self.message)