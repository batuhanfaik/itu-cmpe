# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: person
# @ Date: 12-Dec-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from flask_login import UserMixin


class Student(UserMixin):
    def __init__(self, tr_id, faculty_id, department_id, student_id, semester, grade, gpa,
                 credits_taken, minor):
        self.tr_id = tr_id
        self.faculty_id = faculty_id
        self.department_id = department_id
        self.student_id = student_id
        self.semester = semester
        self.grade = grade
        self.gpa = gpa
        self.credits_taken = credits_taken
        self.minor = minor
