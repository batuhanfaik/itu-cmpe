class Faculty:
    def __init__(self,faculty_name,faculty_shortened_name,departments=[],classrooms=[]):
        self.faculty_id=0
        self.faculty_name=faculty_name
        self.faculty_shortened_name=faculty_shortened_name
        self.departments=departments
        self.classrooms=classrooms
    def set_faculty_id(self,new_faculty_id):
        self.faculty_id=new_faculty_id
    def get_faculty_id(self):
        return self.faculty_id

