import psycopg2 as dbapi2


class Database:
    def __init__(self, dbfile):
        self.dbfile = dbfile
        self.campuses = []
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

    # def add_person(self, person):
    #     tr_id = person.tr_id
    #     self.people[tr_id] = person
    #     return tr_id

    def del_person(self, tr_id):
        if tr_id in self.people:
            del self.people[tr_id]

    def get_person(self, tr_id):
        person = self.people.get(tr_id)
        if person is None:
            return None
        return person

    def get_people(self):
        people = []
        for tr_id, person in self.people.items():
            people.append((tr_id, person))
        return people

    def add_campus(self, campus):
        campus.set_campus_id(self._last_campus_id)
        campus._last_faculty_id = 0
        self.campuses.append(campus)
        self._last_campus_id += 1
        return self._last_campus_id

    def delete_campus(self, campus_id):
        flag = True
        for i in range(len(self.campuses) - 1):
            if flag and (self.campuses[i].get_campus_id() == int(campus_id)):
                del self.campuses[i]
                flag = False
            if not flag:
                self.campuses[i].set_campus_id(self.campuses[i].get_campus_id() - 1)
        if flag:
            if self.campuses[len(self.campuses) - 1].get_campus_id() == campus_id:
                del self.campuses[len(self.campuses) - 1]
        self._last_campus_id = self._last_campus_id - 1
        return self._last_campus_id

    def edit_campus(self, campus):
        for i in range(len(self.campuses)):
            if self.campuses[i].get_campus_id() == campus.get_campus_id():
                self.campuses[i] = campus
                break

    def add_faculty(self, campus_id, faculty):
        faculty.set_faculty_id(self.campuses[campus_id]._last_faculty_id)
        self.campuses[campus_id].add_faculty(faculty, campus_id)
        print(self.campuses)
        return self.campuses[campus_id]._last_faculty_id

    def delete_faculty(self, campus_id, faculty_id):
        flag = True
        for i in range(len(self.campuses[campus_id].faculties) - 1):
            if (flag == True) and (
                    self.campuses[campus_id].faculties[i].get_faculty_id() == int(faculty_id)):
                del self.campuses[campus_id].faculties[i]
                flag = False
            if not flag:
                self.campuses[campus_id].faculties[i].set_faculty_id(
                    self.campuses[campus_id].faculties[i].get_faculty_id() - 1)
        if flag:
            if self.campuses[campus_id].faculties[
                len(self.campuses[campus_id].faculties) - 1].get_faculty_id() == faculty_id:
                del self.campuses[campus_id].faculties[
                    len(self.campuses[campus_id].faculties) - 1]
        self.campuses[campus_id]._last_faculty_id = self.campuses[campus_id]._last_faculty_id - 1
        return self.campuses[campus_id]._last_faculty_id

    def get_campus(self, campus_id):
        campus = self.campuses.get(campus_id)
        if campus is None:
            return None
        campus = campus(campus.name, campus.location)
        return campus

    def get_campuses(self):
        return self.campuses
