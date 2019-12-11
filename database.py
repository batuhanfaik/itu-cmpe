import psycopg2 as dbapi2

from person import Person
from campus import Campus, Faculty


class Database:
    def __init__(self, dbfile):
        self.dbfile = dbfile
        self.campuses = {}
        self._last_campus_id = 0
        self._last_faculty_id = 0
        self.people = {}

    def add_person(self, person):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into people (tr_id, name, surname, phone_number, email, pass, " \
                    "person_category, mother_fname, father_fname, gender, birth_city, birth_date, " \
                    "id_reg_city, id_reg_district) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                   person.email, person.password, person.person_category,
                                   person.mother_fname, person.father_fname, person.gender,
                                   person.birth_city, person.birth_date, person.id_reg_city,
                                   person.id_reg_district))
            connection.commit()
        self.people[person.tr_id] = person
        return person.tr_id

    def update_person(self, person, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update people set tr_id = %s, name = %s, surname = %s, phone_number = %s, " \
                    "email = %s, pass = %s, person_category = %s, mother_fname = %s, " \
                    "father_fname = %s, gender = %s, birth_city = %s, birth_date = %s, " \
                    "id_reg_city = %s, id_reg_district = %s where (tr_id = %s)"
            cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                   person.email, person.password, person.person_category,
                                   person.mother_fname, person.father_fname, person.gender,
                                   person.birth_city, person.birth_date, person.id_reg_city,
                                   person.id_reg_district, tr_id))
            connection.commit

    def delete_person(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from people where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            connection.commit

    def get_person(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from people where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            if(cursor.rowcount == 0):
                return None
        person_ = Person(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return person_

    def get_user(self, username):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from people where (email = %s)"
            if '@' in username:
                cursor.execute(query, (username,))
            else:
                cursor.execute(query, (username + "@itu.edu.tr",))
            if cursor.rowcount == 0:
                return None
        user = Person(*cursor.fetchone())
        if user.person_category == 0:
            user.role = "admin"
        elif user.person_category == 1:
            user.role = "staff"
        elif user.person_category == 2:
            user.role = "instructor"
        else:
            user.role = "student"
        return user

    def get_people(self):
        people = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from people order by name, surname"
            cursor.execute(query)
            for row in cursor:
                person = Person(*row[:])
                people.append((person.tr_id, person))
        return people

    def add_campus(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into campus (name, address, city, size, foundation_date, phone_number,campus_image_name,campus_image_extension,campus_image_data) values (%s, %s, %s, %s, %s,%s,%s ,%s, %s )"
            cursor.execute(query, (campus.name, campus.address,
                                   campus.city, campus.size, campus.foundation_date, campus.phone_number, campus.img_name, campus.img_extension, campus.img_data))
            connection.commit()
        print('End of the campus add function')
        self.campuses[campus.id] = campus
        return campus.id

    def delete_campus(self, campus_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from campus where (id = %s)"
            cursor.execute(query, (campus_id,))
            connection.commit
            # if not flag:
            #     self.campuses[i].set_campus_id(
            #         self.campuses[i].get_campus_id() - 1)
        # if flag:
        #     if self.campuses[len(self.campuses) - 1].get_campus_id() == campus_id:
        #         del self.campuses[len(self.campuses) - 1]
        # self._last_campus_id = self._last_campus_id - 1

    def update_campus(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update campus set name = %s, address = %s, city = %s, size = %s, foundation_date = %s, phone_number = %s, campus_image_name = %s, campus_image_extension = %s, campus_image_data = %s where (id= %s)"
            cursor.execute(query, (campus.name, campus.address, campus.city,
                                   campus.size, campus.foundation_date, campus.phone_number, campus.img_name, campus.img_extension, campus.img_data, campus.id))
            connection.commit

    def update_campus_image(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update campus set image_name = %s, file_extension = %s, image_data = %s where (id= % s)"
            cursor.execute(query, (campus.image_name,
                                   campus.file_extension, campus.image_data, campus.id))
            connection.commit

    def get_campuses(self):
        campuses = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from campus order by id asc"
            cursor.execute(query)
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                campus = Campus(*row[:])
                campuses.append((campus.id, campus))
        return campuses

    def get_campus(self, campus_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from campus where (id = %s)"
            cursor.execute(query, (campus_id,))
            if(cursor.rowcount == 0):
                return None
        campus_ = Campus(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return campus_

    def add_faculty(self, faculty):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (faculty.campus_id, faculty.name, faculty.shortened_name,
                                   faculty.address, faculty.foundation_date, faculty.phone_number))
            connection.commit

    def update_faculty(self, faculty):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update faculty set name = %s, shortened_name = %s, address = %s, foundation_date = %s, phone_number = %s where (id = %s)"
            cursor.execute(query, (faculty.name, faculty.shortened_name, faculty.address,
                                   faculty.foundation_date, faculty.phone_number, faculty.id))
            connection.commit

    def delete_faculty(self, faculty_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from faculty where (id = %s)"
            cursor.execute(query, (faculty_id,))
            connection.commit

    def get_faculties_from_campus(self, campus_id):
        faculties = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from faculty where (campus_id = %s) order by id asc"
            cursor.execute(query, (campus_id,))
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                faculty = Faculty(*row[:])
                faculties.append((faculty.id, faculty))
        return faculties

    def get_faculty(self, faculty_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from faculty where (id = %s)"
            cursor.execute(query, (faculty_id,))
            if(cursor.rowcount == 0):
                return None
        # Inline unpacking of a tuple
        faculty_ = Faculty(*cursor.fetchone()[:])
        return faculty_
