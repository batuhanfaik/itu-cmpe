class Faculty:
    def __init__(self,facultyName,facultyShortenedName,departments=[],classrooms=[]):
        self.facultyid=0
        self.facultyName=facultyName
        self.facultyShortenedName=facultyShortenedName
        self.departments=departments
        self.classrooms=classrooms
    def __setfacultyid__(self,newfacultyid):
        self.facultyid=newfacultyid
    def __getfacultyid__(self):
        return self.facultyid

