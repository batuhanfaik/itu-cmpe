# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: person
# @ Date: 26-Oct-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from flask import current_app
from flask_login import UserMixin

class Person(UserMixin):
    def __init__(self, tr_id, name, surname, phone_number, email, password, person_category,
                 mother_fname, father_fname, gender, birth_city, birth_date, id_reg_city,
                 id_reg_district):
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

        self.active = True
        self.is_admin = False
        if self.person_category == 0:
            self.role = "admin"
            self.is_admin = True
        elif self.person_category == 1:
            self.role = "staff"
        elif self.person_category == 2:
            self.role = "instructor"
        else:
            self.role = "student"




        @property
        def get_id(self):
            return self.email

        @property
        def is_active(self):
            return self.active
