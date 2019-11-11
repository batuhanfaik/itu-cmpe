class Facility:
    def __init__(self,name,shortened_name,size,number_of_workers,expenses,campus_id):
        self.name=name
        self.shortened_name=shortened_name
        self.size=size
        self.number_of_workers=number_of_workers
        self.expenses = expenses
        self.campus_id = campus_id
    def set_id(self,new_id):
        self.id=new_id
    def get_id(self):
        return self.id