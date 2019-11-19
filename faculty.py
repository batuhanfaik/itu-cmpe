class Faculty:
    def __init__(self,faculty_name,faculty_shortened_name,address,foundation_date,phone_number):
        self.faculty_id=0
        self.faculty_name=faculty_name
        self.faculty_shortened_name=faculty_shortened_name
        self.address = address
        self.foundation_date = foundation_date
        self.phone_number = phone_number

    def get_faculty_id(self):
        return self.faculty_id

