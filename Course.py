class Course:
    def __init__(self, crn, start_time, end_time, day, capacity, enrolled, credits,
                 language, classroom_id, faculty_id, instructor_id):
        self.crn = crn
        self.start_time = start_time
        self.end_time = end_time
        self.day = day
        self.capacity = capacity
        self.enrolled = enrolled
        self.credits = credits
        self.language = language
        self.classroom_id = classroom_id
        self.faculty_id = faculty_id
        self.instructor_id = instructor_id