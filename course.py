class Course:
    def __init__(self, crn, code, name, start_time, end_time, day, capacity, enrolled, credits,
                 language, classroom_id, instructor_id, department_id):
        self.crn = crn
        self.code = code
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.day = day
        self.capacity = capacity
        self.enrolled = enrolled
        self.credits = credits
        self.language = language
        self.classroom_id = classroom_id
        self.instructor_id = instructor_id
        self.department_id = department_id
        self.faculty_name = None
        self.department_name = None
        self.instructor_name = None
        self.faculty_name = None
        self.door_number = None


class TakenCourse:
    def __init__(self, id, student_id, crn, datetime, grade):
        self.student_id = student_id
        self.crn = crn
        self.datetime = datetime
        self.id = id
        self.grade = grade

