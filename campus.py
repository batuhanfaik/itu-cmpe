class Campus:
    def __init__(self,name, location,faculties):
        self.name=name
        self.location=location
        self.campusid=0
        self.faculties=faculties
        self._last_faculty_id=len(faculties)
    def __setcampusid__(self,newcampusid):
        self.campusid=newcampusid
    def __getcampusid__(self):
        return self.campusid
    def addFaculty(self,faculty,campusid):
        self.faculties.append(faculty)
        self._last_faculty_id+=1

    