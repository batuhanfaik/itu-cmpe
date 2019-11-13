# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: person
# @ Date: 26-Oct-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Person:
    def __init__(self, tr_id, name, surname, phone_number, email, password, person_category,
                 mother_fname, father_fname, gender, birth_city, birth_date, id_reg_city,
                 id_reg_district):
        assert isinstance(tr_id, str), "TR ID must be a type string"
        assert isinstance(name, str), "Name must be a type string"
        assert isinstance(surname, str), "Surname must be a type string"
        assert isinstance(phone_number, str), "Phone number must be a type string"
        assert isinstance(email, str), "Email must be a type string"
        assert isinstance(password, str), "Password must be a type string"
        assert isinstance(person_category, int), "Person category must be a type integer"
        assert isinstance(mother_fname, str), "Mother first name must be a type string"
        assert isinstance(father_fname, str), "Father first name must be a type string"
        assert isinstance(gender, str), "Gender must be a type string"
        assert isinstance(birth_city, str), "Birth city must be a type string"
        assert isinstance(birth_date, str), "Birth date must be a type string"
        assert isinstance(id_reg_city, str), "ID registration city must be a type string"
        assert isinstance(id_reg_district, str), "ID registration district must be a type string"

        self.tr_id = tr_id
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
