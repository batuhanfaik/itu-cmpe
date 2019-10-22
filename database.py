from campus import Campus

class Database:
    def __init__(self):
        self.campusses=[]
        self._last_campus_id = 0
    def add_campus(self,campus):
        campus.__setcampusid__(self._last_campus_id)
        self.campusses.append(campus)
        self._last_campus_id +=1
        return self._last_campus_id
    def delete_campus(self,campus_id):
        flag=True
        for i in range(len(self.campusses)-1):
            if(flag==True) and (self.campusses[i].__getcampusid__() == int(campus_id)):
                del self.campusses[i]
                flag = False
            if(flag==False):
                self.campusses[i].__setcampusid__(self.campusses[i].__getcampusid__()-1)
        if(flag == True):
            if(self.campusses[len(self.campusses)-1].__getcampusid__()==campus_id):
                del self.campusses[len(self.campusses)-1]
        self._last_campus_id = self._last_campus_id-1
        return self._last_campus_id
    def edit_campus(self,campus):
        for i in range(len(self.campusses)):
            if(self.campusses[i].__getcampusid__() == campus.__getcampusid__()):
                self.campusses[i]=campus
                break   
    def get_campus(self, campus_id):
        campus = self.campusses.get(campus_id)
        if campus is None:
            return None
        campus = campus(campus.name, campus.location)
        return campus

    def get_campusses(self):
        return self.campusses