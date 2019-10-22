from campus import Campus
from faculty import Faculty
class Database:
    def __init__(self):
        self.campusses=[]
        self._last_campus_id = 0
        self._last_faculty_id = 0
    def add_campus(self,campus):
        campus.set_campus_id(self._last_campus_id)
        campus._last_faculty_id=0
        self.campusses.append(campus)
        self._last_campus_id +=1
        return self._last_campus_id

    def delete_campus(self,campus_id):
        flag=True
        for i in range(len(self.campusses)-1):
            if(flag==True) and (self.campusses[i].get_campus_id() == int(campus_id)):
                del self.campusses[i]
                flag = False
            if(flag==False):
                self.campusses[i].set_campus_id(self.campusses[i].get_campus_id()-1)
        if(flag == True):
            if(self.campusses[len(self.campusses)-1].get_campus_id()==campus_id):
                del self.campusses[len(self.campusses)-1]
        self._last_campus_id = self._last_campus_id-1
        return self._last_campus_id

    def edit_campus(self,campus):
        for i in range(len(self.campusses)):
            if(self.campusses[i].get_campus_id() == campus.get_campus_id()):
                self.campusses[i]=campus
                break   

    def add_faculty(self,campus_id,faculty):
        faculty.set_faculty_id(self.campusses[campus_id]._last_faculty_id)
        self.campusses[campus_id].add_faculty(faculty,campus_id)
        print(self.campusses)
        return self.campusses[campus_id]._last_faculty_id

    def delete_faculty(self,campus_id,faculty_id):
        flag=True
        for i in range(len(self.campusses[campus_id].faculties)-1):
            if(flag==True) and (self.campusses[campus_id].faculties[i].get_faculty_id() == int(faculty_id)):
                del self.campusses[campus_id].faculties[i]
                flag = False
            if(flag==False):
                self.campusses[campus_id].faculties[i].set_faculty_id(self.campusses[campus_id].faculties[i].get_faculty_id()-1)
        if(flag):
            if(self.campusses[campus_id].faculties[len(self.campusses[campus_id].faculties)-1].get_faculty_id()==faculty_id):
                del self.campusses[campus_id].faculties[len(self.campusses[campus_id].faculties)-1]
        self.campusses[campus_id]._last_faculty_id = self.campusses[campus_id]._last_faculty_id-1
        return self.campusses[campus_id]._last_faculty_id

    def get_campus(self, campus_id):
        campus = self.campusses.get(campus_id)
        if campus is None:
            return None
        campus = campus(campus.name, campus.location)
        return campus
    
    def get_campusses(self):
        return self.campusses

    