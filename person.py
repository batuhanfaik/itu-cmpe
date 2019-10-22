# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: person
# @ Date: 26-Oct-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Person:
    def __init__(self, tr_id, name, surname):
        assert isinstance(tr_id, str), "TR ID must be a type string"
        assert isinstance(name, str), "Name must be a type string"
        assert isinstance(surname, str), "Surname must be a type string"

        self.tr_id = tr_id
        self.name = name
        self.surname = surname
