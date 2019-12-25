Parts Implemented by Fatih Altınpınar
=====================================

Tables
------


.. code-block:: sql
    :linenos:
    :caption: SQL of Course Table

    CREATE TABLE IF NOT EXISTS COURSE (
        crn             CHAR(5)     NOT NULL PRIMARY KEY,
        code            CHAR(3)     NOT NULL,
        name            VARCHAR(100) NOT NULL,
        start_time      TIME        NOT NULL,
        end_time        TIME        NOT NULL,
        day             VARCHAR(9)  NOT NULL,
        capacity        INT         NOT NULL,
        enrolled        INT         default(0),
        credits         REAL        NOT NULL,
        language        CHAR(2)     default('en'),
        classroom_id    INT         NOT NULL,
        instructor_id   INT         NOT NULL,
        department_id   INT         NOT NULL,
        info            TEXT        NULL,
        FOREIGN KEY (classroom_id) REFERENCES CLASSROOM (id) on delete cascade on update cascade,
        FOREIGN KEY (instructor_id) REFERENCES INSTRUCTOR (id) on delete cascade on update cascade,
        FOREIGN KEY (department_id) REFERENCES DEPARTMENT (id) on delete cascade on update cascade
    );



.. code-block:: sql
    :linenos:
    :caption: SQL of Classroom Table

    CREATE TABLE IF NOT EXISTS CLASSROOM(
        id              SERIAL      NOT NULL PRIMARY KEY,
        capacity        INT         NOT NULL,
        has_projection  BOOLEAN     DEFAULT false,
        door_number     VARCHAR(4)     NOT NULL,
        floor           VARCHAR(2),
        renewed         BOOLEAN     DEFAULT false,
        board_count     INT,
        air_conditioner BOOLEAN     DEFAULT false,
        faculty_id      INT         NOT NULL,
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id) on delete cascade on update cascade,
        unique(door_number, faculty_id)
    );


.. code-block:: sql
    :linenos:
    :caption: SQL of Taken Course Table

    CREATE TABLE IF NOT EXISTS TAKEN_COURSE(
        id SERIAL PRIMARY KEY,
        student_id BIGINT NOT NULL,
        crn CHAR(5) NOT NULL,
        grade REAL NULL,
        datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES STUDENT (student_id)  on delete cascade on update cascade,
        FOREIGN KEY (crn) REFERENCES COURSE (crn)  on delete cascade on update cascade,
        UNIQUE(student_id, crn),
        CHECK ( grade >= 0 and grade <= 4 )
    );



.. code-block:: sql
    :linenos:
    :caption: SQL of Instructor Table

    CREATE TABLE IF NOT EXISTS INSTRUCTOR(
        id SERIAL NOT NULL PRIMARY KEY,
        tr_id BIGINT NOT NULL,
        department_id INT NOT NULL,
        faculty_id INT NOT NULL,
        specialization VARCHAR(80),
        bachelors VARCHAR(80),
        masters VARCHAR(80),
        doctorates VARCHAR(80),
        room_id CHAR(4),
        FOREIGN KEY (tr_id) REFERENCES PEOPLE (tr_id)  on delete cascade on update cascade,
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id) on delete cascade on update cascade,
        FOREIGN KEY (department_id) REFERENCES DEPARTMENT (id) on delete cascade on update cascade,
        unique(tr_id)
    );


.. code-block:: sql
    :linenos:
    :caption: SQL of Syllabus Table

    CREATE TABLE IF NOT EXISTS SYLLABUS (
        crn             char(5)         PRIMARY KEY,
        file            bytea           default null,
        foreign key (crn) references course(crn) on delete cascade on update cascade
    );

Database Queries
----------------

Database queries are handled via ``database.py`` file by constructing a Database class and using ``psycopg2`` library as the PostgreSQL driver.

Below are the related class methods implemented by this member:

Course
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Course Table
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

Syllabus
++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Syllabus Table
    def add_syllabus(self, crn, syllabus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """insert into syllabus (crn, file) values (%s, %s);"""
            cursor.execute(query, (crn, syllabus))
        pass

    def update_syllabus(self, crn, syllabus):
        self.delete_syllabus(crn)
        self.add_syllabus(crn, syllabus)

    def delete_syllabus(self, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """delete from syllabus where crn = %s;"""
            cursor.execute(query, (crn, ))
        pass

    def get_syllabus(self, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """select file from syllabus where crn = %s;"""
            cursor.execute(query, (crn,))
            if cursor.rowcount == 0:
                return None
            syllabus = cursor.fetchone()
            return syllabus

Classroom
+++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Classroom Table
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

Instructor
++++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Instructor Table
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

Taken Course
++++++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Taken Course Table

    def add_taken_course(self, student_id, crn):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = """insert into taken_course (student_id, crn) values (%s, %s);"""
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


Templates
---------

Following templates are written by **this user**:
    - ``edit_classroom.html``
    - ``edit_course.html``
    - ``edit_instructor.html``
    - ``instructors.html``
    - ``courses.html``
    - ``select_courses.html``


Following templates are written both by **this member** and **other teammates**:
    - ``layout.html``
    - ``login.html``
