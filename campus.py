class Campus:
    def __init__(self, name, location, faculties):
        self.campus_id = 0
        self.name = name
        self.location = location
        self.faculties = faculties
        self._last_faculty_id = len(faculties)

    def set_campus_id(self, new_campus_id):
        self.campus_id = new_campus_id

    def get_campus_id(self):
        return self.campus_id

    def add_faculty(self, faculty, campus_id):
        self.faculties.append(faculty)
        self._last_faculty_id += 1
