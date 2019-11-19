class Campus:
    def __init__(self, name, adress,city,size,phone_number):
        self.campus_id = 0
        self.name = name
        self.address = adress
        self.city = city
        self.size = size
        self.phone_number = phone_number
        
    def get_campus_id(self):
        return self.campus_id

    def add_faculty(self, faculty, campus_id):
        self.faculties.append(faculty)
        self._last_faculty_id += 1

    def remove_faculty(self, faculty_id,campus_id):
        self.faculties
