# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @ Filename: assistant
# @ Date: 13-Dec-2019
# @ AUTHOR: batuhanfaik
# @ Copyright (C) 2019 Batuhan Faik Derinbay
# @ Project: itucsdb1922
# @ Description: Not available
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from flask_login import UserMixin


class Assistant(UserMixin):
    def __init__(self, tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa,
                 research_area, office_day, office_hour_start, office_hour_end):
        self.tr_id = tr_id
        self.faculty_id = faculty_id
        self.supervisor = supervisor
        self.assistant_id = assistant_id
        self.bachelors = bachelors
        self.degree = degree
        self.grad_gpa = grad_gpa
        self.research_area = research_area
        self.office_day = office_day
        self.office_hour_start = office_hour_start
        self.office_hour_end = office_hour_end
