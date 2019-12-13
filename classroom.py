class Classroom:
    def __init__(self, id, capacity, has_projection, door_number, floor, renewed,
                 board_count, air_conditioner, faculty_id):
        self.renewed = renewed
        self.air_conditioner = air_conditioner
        self.faculty_id = faculty_id
        self.board_count = board_count
        self.floor = floor
        self.door_number = door_number
        self.id = id
        self.capacity = capacity
        self.has_projection = has_projection
