class Facility:
    def __init__(self,id,campus_id,name,shortened_name,number_of_workers, size,expenses):
        self.id=id
        self.name = name
        self.shortened_name=shortened_name
        self.size=size
        self.number_of_workers=number_of_workers
        self.expenses = expenses
        self.campus_id = campus_id

