class Campus:
    def __init__(self, campus_id, name, adress, city, size, foundation_date, phone_number, image_name, image_data):
        self.id = campus_id
        self.name = name
        self.address = adress
        self.city = city
        self.size = size
        self.foundation_date = foundation_date
        self.phone_number = phone_number
        self.img_name = image_name
        self.img_data = image_data

    def get_campus_id(self):
        return self.id
