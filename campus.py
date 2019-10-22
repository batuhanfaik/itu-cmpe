class Campus:
    def __init__(self,name, location):
        self.name=name
        self.location=location
        self.campusid=0
    def __setcampusid__(self,newcampusid):
        self.campusid=newcampusid
    def __getcampusid__(self):
        return self.campusid