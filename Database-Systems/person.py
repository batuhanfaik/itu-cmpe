# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: person
# @ Date: 26-Oct-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from base64 import b64encode

from flask_login import UserMixin


class Person(UserMixin):
    def __init__(self, tr_id, name, surname, phone_number, email, password, person_category,
                 mother_fname, father_fname, gender, birth_city, birth_date, id_reg_city,
                 id_reg_district, photo_name, photo_extension, photo_data):
        self.tr_id = tr_id
        self.id = email
        self.name = name
        self.surname = surname
        self.phone_number = phone_number
        self.email = email
        self.password = password
        self.person_category = person_category
        self.mother_fname = mother_fname
        self.father_fname = father_fname
        self.gender = gender
        self.birth_city = birth_city
        self.birth_date = birth_date
        self.id_reg_city = id_reg_city
        self.id_reg_district = id_reg_district
        self.photo_name = photo_name
        self.photo_extension = photo_extension
        self.photo_data = photo_data

        if self.photo_data:
            self.photo = b64encode(self.photo_data)
            self.photo = self.photo.decode('utf-8')

        self.active = True
        if self.person_category == 0:
            self.is_admin = True
        else:
            self.is_admin = False

        @property
        def get_id(self):
            return self.email

        @property
        def is_active(self):
            return self.active
