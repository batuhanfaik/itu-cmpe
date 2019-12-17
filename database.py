import psycopg2 as dbapi2
from psycopg2 import Error

from assistant import Assistant
from campus import Campus, Faculty, Department
from person import Person
from student import Student
from instructor import Instructor
from staff import Staff
from classroom import Classroom
from course import Course
from course import TakenCourse
from facility import Facility
from staff_facil import Staff_facil

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

    # faati's cruds #
    # taken_course crud#
    def add_taken_course(self, student_id, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """insert into taken_course (student_id, crn) values (%s, %s)"""
            cursor.execute(query, (student_id, crn))
        pass

    def update_taken_course(self, id, takencourse):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """update taken_course set student_id = %s, crn = %s, grade = %s
                        where (id = %s)"""
            cursor.execute(query, (takencourse.student_id, takencourse.crn, takencourse.grade, id))
        return id

    def delete_taken_course(self, student_id, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """delete from taken_course where (student_id = %s and crn = %s)"""
            cursor.execute(query, (student_id, crn))

    def get_taken_course(self, id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select * from taken_course where (id = %s)"""
            cursor.execute(query, (id,))
            return TakenCourse(*cursor.fetchone)

    def get_taken_course_by_crn(self, crn):
        students = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from taken_course where (crn = %s)"
            cursor.execute(query, (crn,))
            for row in cursor:
                taken_course = TakenCourse(*row[:])
                students.append(taken_course)
        return students

    def get_courses_taken_by_student(self, student_id):
        courses = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select course.*, faculty.shortened_name, department.shortened_name,
                            people.name, people.surname, classroom.door_number, taken_course.grade
                            from course, classroom, faculty, instructor, department, people, taken_course
                            where (course.department_id = department.id
                            and course.instructor_id = instructor.id
                            and classroom.faculty_id = faculty.id
                            and course.classroom_id = classroom.id
                            and people.tr_id = instructor.tr_id
                            and taken_course.crn = course.crn
                            and student_id = %s) order by (course.crn);"""
            cursor.execute(query, (student_id,))
            for row in cursor:
                course = Course(*row[:14])
                course.faculty_name = row[14]
                course.department_name = row[15]
                course.instructor_name = row[16] + " " + row[17]
                course.door_number = row[18]
                course.grade = row[19]
                courses.append(course)
        return courses

    # instructor crud #
    def add_instructor(self, instructor):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "INSERT INTO INSTRUCTOR (tr_id, department_id, faculty_id, specialization," \
                    " bachelors, masters, doctorates, room_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(query, (instructor.tr_id, instructor.department_id, instructor.faculty_id,
                                   instructor.specialization, instructor.bachelors, instructor.masters,
                                   instructor.doctorates, instructor.room_id))
        pass

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
        instructor = Instructor(*cursor.fetchone())  # Inline unpacking of a tuple
        return instructor
    def get_instructor_via_tr_id(self, tr_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from instructor where (tr_id = %s)"
            cursor.execute(query, (tr_id,))
            if cursor.rowcount == 0:
                return None
        instructor = Instructor(*cursor.fetchone())  # Inline unpacking of a tuple
        return instructor
    def get_all_instructors(self):
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

    def is_instructor_available(self, start_time, end_time, day, instructor_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select * from course where (instructor_id = %s 
                                   and course.day = %s
                                    and not (( %s < start_time and %s < start_time)
                                    or (%s > end_time and %s > end_time)));"""
            cursor.execute(query, (instructor_id, day, start_time, end_time, start_time, end_time))
            if cursor.rowcount > 0:
                return False
        return True
    # classroom crud #

    def add_classroom(self, classroom):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """insert into classroom (capacity, has_projection, 
                        door_number, floor, renewed, board_count, air_conditioner,
                        faculty_id) values (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(query, (classroom.capacity, classroom.has_projection, classroom.door_number,
                                   classroom.floor, classroom.renewed, classroom.board_count,
                                   classroom.air_conditioner, classroom.faculty_id))
            pass

    def update_classroom(self, id, classroom):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """update classroom set capacity = %s, has_projection = %s, door_number = %s, floor = %s,
                        renewed = %s, board_count = %s, air_conditioner = %s, faculty_id = %s where (id = %s)"""
            cursor.execute(query, (classroom.capacity, classroom.has_projection, classroom.door_number,
                                   classroom.floor, classroom.renewed, classroom.board_count,
                                   classroom.air_conditioner, classroom.faculty_id, id))

        return classroom.id

    def delete_classroom(self, id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from classroom where (id = %s)"
            cursor.execute(query, (id,))
        pass

    def get_classroom(self, id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from classroom where (id = %s)"
            cursor.execute(query, (id,))
            if cursor.rowcount == 0:
                return None
        classroom = Classroom(*cursor.fetchone())  # Inline unpacking of a tuple
        return classroom

    def get_classroom_by_door_and_faculty(self, faculty_id, door_number):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from classroom where(faculty_id = %s and door_number = %s);"
            cursor.execute(query, (faculty_id, door_number))
            if cursor.rowcount == 0:
                return None
            return Classroom(*cursor.fetchone())

    def get_all_classrooms(self):
        classrooms = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute("select * from classroom order by (id);")
            for row in cursor:
                classrooms.append(Classroom(*row))
        return classrooms
    
    def get_all_classrooms_by_faculty(self, faculty_id):
        classrooms = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute("select * from classroom where (faculty_id = %s) order by (id);", (faculty_id,))
            for row in cursor:
                classrooms.append(Classroom(*row))
        return classrooms

    def is_classroom_available(self, start_time, end_time, day, classroom_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select * from course where (classroom_id = %s 
                        and course.day = %s
                        and not (( %s < start_time and %s < start_time)
                                or (%s > end_time and %s > end_time)));"""
            cursor.execute(query, (classroom_id, day, start_time, end_time, start_time, end_time))
            if cursor.rowcount > 0:
                return False
        return True

    # course crud #
    def add_course(self, course):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """insert into course (crn, code, name, start_time, end_time, day, capacity, enrolled,
                        credits, language, classroom_id , instructor_id, department_id, info)
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(query, (course.crn, course.code, course.name, course.start_time, course.end_time,
                                   course.day, course.capacity, course.enrolled, course.credits, course.language,
                                   course.classroom_id, course.instructor_id, course.department_id, course.info))
        pass

    def update_course(self, crn, course):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """update course set crn = %s, code = %s, name = %s, start_time = %s, end_time = %s,
                        day = %s, capacity = %s, enrolled = %s, credits = %s, language = %s, classroom_id = %s, 
                        instructor_id = %s, department_id = %s, info = %s where (crn = %s)"""
            cursor.execute(query, (crn, course.code, course.name, course.start_time, course.end_time,
                                   course.day, course.capacity, course.enrolled, course.credits, course.language,
                                   course.classroom_id, course.instructor_id,
                                   course.department_id, course.info, crn))
        return course.crn

    def delete_course(self, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from course where (crn = %s)"
            cursor.execute(query, (crn,))

        pass

    def get_course(self, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from course where (crn = %s)"
            cursor.execute(query, (crn,))
            if cursor.rowcount == 0:
                return None
        course = Course(*cursor.fetchone())
        return course

    def get_course_via_instructor_id(self, instructor_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from course where (instructor_id = %s)"
            cursor.execute(query, (instructor_id,))
            if cursor.rowcount == 0:
                return None
        course = Course(*cursor.fetchone())
        return course

    def get_courses_by_instructor_id(self, instructor_id):
        courses = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select course.*, faculty.shortened_name, department.shortened_name,
                            people.name, people.surname, classroom.door_number
                            from course, classroom, faculty, instructor, department, people
                            where (course.department_id = department.id
                            and course.instructor_id = instructor.id
                            and classroom.faculty_id = faculty.id
                            and course.classroom_id = classroom.id
                            and people.tr_id = instructor.tr_id
                            and course.instructor_id = %s) order by (course.crn);"""
            cursor.execute(query, (instructor_id,))
            for row in cursor:
                course = Course(*row[:14])
                course.faculty_name = row[14]
                course.department_name = row[15]
                course.instructor_name = row[16] + " " + row[17]
                course.door_number = row[18]
                courses.append(course)
        return courses

    def get_all_courses(self):
        courses = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute("""select course.*, faculty.shortened_name, department.shortened_name,
                            people.name, people.surname, classroom.door_number
                            from course, classroom, faculty, instructor, department, people
                            where (course.department_id = department.id
                            and course.instructor_id = instructor.id
                            and classroom.faculty_id = faculty.id
                            and course.classroom_id = classroom.id
                            and people.tr_id = instructor.tr_id) order by (department.shortened_name);""")
            for row in cursor:
                course = Course(*row[:14])
                course.faculty_name = row[14]
                course.department_name = row[15]
                course.instructor_name = row[16] + " " + row[17]
                course.door_number = row[18]
                courses.append(course)
        return courses

    def update_course_enrollment(self, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute("""select count(student_id) from taken_course where crn = %s;""", (crn,))
            number = cursor.fetchone()
            cursor.execute("""update course set enrolled = %s where crn = %s""", (number, crn))
        return number

    def student_can_take_course(self, student_id, course):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select * from course, taken_course where (course.crn = taken_course.crn
                        and taken_course.student_id = %s
                        and course.crn <> %s 
                        and course.day = %s
                        and not (( %s < start_time and %s < start_time)
                                or (%s > end_time and %s > end_time)))"""
            cursor.execute(query, (student_id, course.crn, course.day, course.start_time,
                                   course.end_time, course.start_time, course.end_time))
            if cursor.rowcount > 0:
                return False
            return True

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
        user.instructor_id = None
        user.student_id = None
        if user.person_category == 0:
            user.role = "admin"
        elif user.person_category == 1:
            user.role = "staff"
        elif user.person_category == 2:
            user.role = "instructor"
            try:
                user.instructor_id = self.get_instructor_via_tr_id(user.tr_id).id
            except Error as e:
                print(e)
        else:
            user.role = "student"
            try:
                user.student_id = self.get_student(user.tr_id).student_id
            except Error as e:
                print(e)

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
            query = "insert into staff (id, manager_name, absences, hire_date, authority_lvl,department, social_sec_no) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (staff.id, staff.manager_name, staff.absences, staff.hire_date, staff.authority_lvl, staff.department,
                                   staff.social_sec_no))
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
    def delete_staff(self,staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from staff where (id = %s)"
            cursor.execute(query, (staff_id,))
            connection.commit

    def update_staff(self,staff):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update staff set  manager_name = %s, absences = %s, hire_date = %s, authority_lvl = %s,department = %s, social_sec_no = %s where (id = %s)"
            cursor.execute(query, ( staff.manager_name, staff.absences, staff.hire_date, staff.authority_lvl, staff.department,
                                   staff.social_sec_no, staff.id))
            connection.commit

    def delete_staff_facil(self,staff_id, facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from staff_facil where (staff_id= %s and facility_id = %s)"
            cursor.execute(query, (staff_id,facility_id))
            connection.commit

    def update_SF(self,SF):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update staff_facil set  title = %s, from_date = %s, to_date= %s, salary = %s, duty = %s where (staff_id = %s and facility_id = %s)"
            cursor.execute(query, ( staff_facil.title, staff_facil.from_date, staff_facil.to_date, staff_facil.salary,
                                   staff_facil.duty, staff_facil.staff_id, staff_facil.facility_id))
            connection.commit

    def get_facility(self,facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility where (id = %s)"
            cursor.execute(query, (facility_id,))
            if (cursor.rowcount == 0):
                return None
        found_facility = Facility(*cursor.fetchone()[:])
        return found_facility
    def get_all_facility(self):
        all_facility = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility order by id asc"
            cursor.execute(query)
            for row in cursor:
                facil = Facility(*row[:])
                all_facility.append(facil)
        return all_facility
    def delete_facility(self,facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from facility where (id = %s)"
            cursor.execute(query, (facility_id,))
            connection.commit
    def add_facility(self,facility):
        with dbapi2.connect(self.dbfile) as connection:
            print("TRYİNG TO ADD:")
            print("----------")
            cursor = connection.cursor()
            query = "insert into facility (id, campus_id, name, shortened_name,number_of_workers,size,expenses) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (facility.id, facility.campus_id, facility.name, facility.shortened_name,
                                   facility.number_of_workers, facility.size, facility.expenses))
            connection.commit
    def get_facility_from_campus(self, campus_id):
        facilities = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility where (campus_id = %s) order by id asc"
            cursor.execute(query, (campus_id,))
            for row in cursor:
                facility = Facility(*row[:])
                facilities.append(facility)
        return facilities

    def get_facility_from_staff(self, staff_id):
        staff_facilities = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff_facil where (staff_id = %s) order by staff_id asc"
            cursor.execute(query, (staff_id,))
            for row in cursor:
                SF = Staff_facil(*row[:])
                staff_facilities.append(SF)
        return staff_facilities
    def get_a_facility_from_staff(self, staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff_facil where (staff_id = %s) order by staff_id asc"
            cursor.execute(query, (staff_id,))
            connection.commit

    def add_staff_facility(self,staff_facil):
        with dbapi2.connect(self.dbfile) as connection:
            print("TRYİNG TO ADD:")
            print("----------")
            cursor = connection.cursor()
            query = "insert into staff_facil (title,from_date,to_date,salary,facility_id,staff_id,duty) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (staff_facil.title, staff_facil.from_date, staff_facil.to_date, staff_facil.salary,
                                   staff_facil.facility_id, staff_facil.staff_id, staff_facil.duty))
            connection.commit





    def update_facility(self,facility):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update facility set  id = %s, campus_id = %s, name = %s, shortened_name = %s,number_of_workers = %s, size = %s, expenses = %s where (id = %s)"
            cursor.execute(query, (facility.id, facility.campus_id, facility.name, facility.shortened_name, facility.number_of_workers,
                                   facility.size,
                                   facility.expenses))
            connection.commit
