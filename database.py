import psycopg2 as dbapi2

from assistant import Assistant
from campus import Campus, Faculty, Department
from person import Person
from student import Student
from instructor import Instructor
from staff import Staff


class Database:
    def __init__(self, dbfile):
        self.dbfile = dbfile
        self.campuses = {}
        self._last_campus_id = 0
        self._last_faculty_id = 0
        self.people = {}
        self.students = {}
        self.assistants = {}
        self.staffs = {}

    ### faati's cruds ###

    ### instructor table ###
    def add_instructor(self, instructor):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "INSERT INTO INSTRUCTOR (tr_id, department_id, faculty_id, specialization," \
                    " bachelors, masters, doctorates, room_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(query, (instructor.tr_id, instructor.department_id, instructor.faculty_id,
                                   instructor.specialization, instructor.bachelors, instructor.masters,
                                   instructor.doctorates, instructor.room_id))

        return instructor.id

    def update_instructor(self, id, instructor):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update instructor set tr_id = %s, department_id = %s, faculty_id = %s," \
                    "specialization = %s, bachelors = %s, masters = %s, doctorates = %s, room_id = %s where (id = %s)"
            cursor.execute(query, (instructor.tr_id, instructor.department_id, instructor.faculty_id,
                           instructor.specialization, instructor.bachelors, instructor.masters, instructor.doctorates,
                           instructor.room_id, id))

        return instructor.id

    def delete_instructor(self, id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from instructor where (id = %s)"
            cursor.execute(query, (id,))

        pass

    def get_instructor(self, id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from instructor where (id = %s)"
            cursor.execute(query, (id,))
            if cursor.rowcount == 0:
                return None
        instructor  = Instructor(*cursor.fetchone())  # Inline unpacking of a tuple
        return instructor

    def get_instructors(self):
        instructors = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute("select instructor.*, people.name, people.surname, department.name, faculty.name "
                           "from people, instructor, department, faculty "
                           "where (people.tr_id = instructor.tr_id "
                           "and instructor.department_id = department.id "
                           "and instructor.faculty_id = faculty.id);")
            for row in cursor:
                instructor = Instructor(*row[:9])
                instructor.name = row[9]
                instructor.surname = row[10]
                instructor.departmentName = row[11]
                instructor.facultyName = row[12]
                instructors.append(instructor)
        return instructors


    ########################
    def add_person(self, person):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into people (tr_id, name, surname, phone_number, email, pass, " \
                    "person_category, mother_fname, father_fname, gender, birth_city, birth_date, " \
                    "id_reg_city, id_reg_district, photo_name, photo_extension, photo_data) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                   person.email, person.password, person.person_category,
                                   person.mother_fname, person.father_fname, person.gender,
                                   person.birth_city, person.birth_date, person.id_reg_city,
                                   person.id_reg_district, person.photo_name,
                                   person.photo_extension, person.photo_data))
            connection.commit()
        self.people[person.tr_id] = person
        return person.tr_id

    def update_person(self, person, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update people set tr_id = %s, name = %s, surname = %s, phone_number = %s, " \
                    "email = %s, pass = %s, person_category = %s, mother_fname = %s, " \
                    "father_fname = %s, gender = %s, birth_city = %s, birth_date = %s, " \
                    "id_reg_city = %s, id_reg_district = %s, photo_name = %s, photo_extension = %s, photo_data = %s where (tr_id = %s)"
            cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                   person.email, person.password, person.person_category,
                                   person.mother_fname, person.father_fname, person.gender,
                                   person.birth_city, person.birth_date, person.id_reg_city,
                                   person.id_reg_district, person.photo_name,
                                   person.photo_extension, person.photo_data, tr_id))
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
            if (cursor.rowcount == 0):
                return None
        person_ = Person(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return person_

    def get_person_email(self, email):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from people where (email = %s)"
            cursor.execute(query, (email,))
            if (cursor.rowcount == 0):
                return None
        person_ = Person(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return person_

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

    def add_student(self, student):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into student (tr_id, faculty_id, department_id, student_id, semester, grade, gpa, credits_taken, minor) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (
                student.tr_id, student.faculty_id, student.department_id, student.student_id,
                student.semester, student.grade, student.gpa, student.credits_taken, student.minor))
            connection.commit()
        self.students[student.tr_id] = student
        return student.tr_id

    def update_student(self, student, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update student set tr_id = %s, faculty_id = %s, department_id = %s, student_id = %s, " \
                    "semester = %s, grade = %s, gpa = %s, credits_taken = %s, " \
                    "minor = %s where (tr_id = %s)"
            cursor.execute(query, (
                student.tr_id, student.faculty_id, student.department_id, student.student_id,
                student.semester, student.grade, student.gpa, student.credits_taken, student.minor,
                tr_id))
            connection.commit

    def delete_student(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from student where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            connection.commit

    def get_student(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from student where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            if (cursor.rowcount == 0):
                return None
        student_ = Student(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return student_

    def get_student_via_student_id(self, student_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from student where (student_id = %s)"
            cursor.execute(query, (student_id,))
            if (cursor.rowcount == 0):
                return None
        student_ = Student(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return student_

    def get_students(self):
        students = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from student order by student_id"
            cursor.execute(query)
            for row in cursor:
                student = Student(*row[:])
                students.append((student.tr_id, student))
        return students

    def add_assistant(self, assistant):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (
                assistant.tr_id, assistant.faculty_id, assistant.supervisor, assistant.assistant_id,
                assistant.bachelors, assistant.degree, assistant.grad_gpa, assistant.research_area,
                assistant.office_day, assistant.office_hour_start, assistant.office_hour_end))
            connection.commit()
        self.assistants[assistant.tr_id] = assistant
        return assistant.tr_id

    def update_assistant(self, assistant, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update assistant set tr_id = %s, faculty_id = %s, supervisor = %s, assistant_id = %s, " \
                    "bachelors = %s, degree = %s, grad_gpa = %s, research_area = %s, office_day = %s, office_hour_start = %s, " \
                    "office_hour_end = %s where (tr_id = %s)"
            cursor.execute(query, (
                assistant.tr_id, assistant.faculty_id, assistant.supervisor, assistant.assistant_id,
                assistant.bachelors, assistant.degree, assistant.grad_gpa, assistant.research_area,
                assistant.office_day, assistant.office_hour_start, assistant.office_hour_end,
                tr_id))
            connection.commit

    def delete_assistant(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from assistant where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            connection.commit

    def get_assistant(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from assistant where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            if (cursor.rowcount == 0):
                return None
        assistant_ = Assistant(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return assistant_

    def get_assistant_via_assistant_id(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from assistant where (assistant_id = %s)"
            cursor.execute(query, (tr_id,))
            if (cursor.rowcount == 0):
                return None
        assistant_ = Assistant(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return assistant_

    def get_assistants(self):
        assistants = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from assistant order by assistant_id"
            cursor.execute(query)
            for row in cursor:
                assistant = Assistant(*row[:])
                assistants.append((assistant.tr_id, assistant))
        return assistants

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

    def add_campus(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into campus (name, address, city, size, foundation_date, phone_number,campus_image_extension,campus_image_data) values (%s, %s, %s, %s, %s,%s,%s ,%s)"
            cursor.execute(query, (campus.name, campus.address,
                                   campus.city, campus.size, campus.foundation_date,
                                   campus.phone_number,campus.img_extension,
                                   campus.img_data))
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
            print('hey')
            cursor = connection.cursor()
            query = "update campus set name = %s, address = %s, city = %s, size = %s, foundation_date = %s, phone_number = %s,campus_image_extension = %s, campus_image_data = %s where (id= %s)"
            cursor.execute(query, (campus.name, campus.address, campus.city,
                                   campus.size, campus.foundation_date, campus.phone_number, campus.img_extension, campus.img_data,
                                   campus.id))
            connection.commit

    def update_campus_image(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update campus set file_extension = %s, image_data = %s where (id= % s)"
            cursor.execute(query, (campus.file_extension, campus.image_data, campus.id))
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
            if (cursor.rowcount == 0):
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
            if (cursor.rowcount == 0):
                return None
        # Inline unpacking of a tuple
        faculty_ = Faculty(*cursor.fetchone()[:])
        return faculty_

    def add_department(self, department):
        print('Enter add department')
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            print(department)
            query = "insert into department (faculty_id, name, shortened_name, block_number, budget, foundation_date, phone_number) values (%s, %s, %s,%s,%s,%s,%s)"
            cursor.execute(query,
                           (department.faculty_id, department.name, department.shortened_name,
                            department.block_number, department.budget, department.foundation_date,
                            department.phone_number))
            connection.commit

    def update_department(self, department):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update department set name = %s, shortened_name = %s, block_number = %s, budget = %s, foundation_date = %s, phone_number = %s where (id = %s)"
            cursor.execute(query, (
                department.name, department.shortened_name, department.block_number,
                department.budget,
                department.foundation_date, department.phone_number, department.id,))
            connection.commit

    def delete_department(self, department_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from department where (id = %s)"
            cursor.execute(query, (department_id,))
            connection.commit

    def get_departments_from_faculty(self, faculty_id):
        departments = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from department where (faculty_id = %s) order by id asc"
            cursor.execute(query, (faculty_id,))
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                department = Department(*row[:])
                departments.append((department.id, department))
        return departments

    def get_department(self, department_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from department where (id = %s)"
            cursor.execute(query, (department_id,))
            if (cursor.rowcount == 0):
                return None
        # Inline unpacking of a tuple
        department_ = Department(*cursor.fetchone()[:])
        return department_

    def add_staff(self,staff):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into staff (id, manager_name, absences, hire_date, social_sec_no, authority_lvl,department) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (staff.id, staff.manager_name, staff.absences, staff.hire_date,
                                   staff.social_sec_no, staff.authority_lvl, staff.department))
            connection.commit

    def get_staff(self,staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff where (id = %s)"
            cursor.execute(query, (staff_id,))
            if (cursor.rowcount == 0):
                return None
        found_staff = Staff(*cursor.fetchone()[:])
        return found_staff
    def get_all_staff(self):
        all_staff = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff order by id asc"
            cursor.execute(query)
            for row in cursor:
                staf = Staff(*row[:])
                all_staff.append(staf)
        return all_staff
