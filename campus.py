class Campus:
    def __init__(self, campus_id, name, address, city, size, foundation_date, phone_number, image_name, image_extension, image_data):
        self.id = campus_id
        self.name = name
        self.address = address
        self.city = city
        self.size = size
        self.foundation_date = foundation_date
        self.phone_number = phone_number
        self.img_name = image_name
        self.img_extension = image_extension
        self.img_data = image_data

    def get_campus_id(self):
        return self.id


class Faculty:
    def __init__(self, faculty_id, campus_id, name, shortened_name, adress, foundation_date, phone_number):
        self.id = faculty_id
        self.campus_id = campus_id
        self.name = name
        self.shortened_name = shortened_name
        self.address = adress
        self.foundation_date = foundation_date
        self.phone_number = phone_number

    def get_faculty_id(self):
        return self.id


class Department:
    def __init__(self, department_id, faculty_id, name, shortened_name, adress, foundation_date, phone_number):
        self.id = department_id
        self.faculty_id = faculty_id
        self.name = name
        self.shortened_name = shortened_name
        self.block_number = block_number
        self.budget = budget
        self.foundation_date = foundation_date
        self.phone_number = phone_number

    def get_department_id(self):
        return self.id
