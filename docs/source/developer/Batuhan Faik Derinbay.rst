Parts Implemented by Batuhan Faik Derinbay
==========================================

Tables
------

Tables implemented by this user can be examined in detailed with the given source code from the ``dbinit.py`` file.

Tables People, Assistant and Student are created and implemented by this user and their corresponding statements are given below.

.. code-block:: sql
    :linenos:
    :caption: SQL of People Table

    CREATE TABLE IF NOT EXISTS PEOPLE (
            tr_id BIGINT PRIMARY KEY NOT NULL,
            name VARCHAR(40) NOT NULL,
            surname VARCHAR(40) NOT NULL,
            phone_number VARCHAR(20) not null,
            email VARCHAR(60) NOT NULL,
            pass varchar(256) not null,
            person_category SMALLINT NOT NULL,
            mother_fname varchar(40) null,
            father_fname varchar(40) null,
            gender char(1) null,
            birth_city varchar(50) null,
            birth_date date not null,
            birth_date varchar(50) not null,
            id_reg_district varchar(50) not null,
            photo_name varchar(256),
            photo_extension varchar(10),
            photo_data bytea,
            unique (tr_id),
            unique (email)
        );

.. code-block:: sql
    :linenos:
    :caption: SQL of Assistant Table

    CREATE TABLE IF NOT EXISTS ASSISTANT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id) on delete cascade on update cascade NOT NULL,
        faculty_id int references FACULTY(id) on delete cascade on update cascade not null,
        supervisor bigint references INSTRUCTOR(tr_id) on delete cascade on update cascade not null,
        assistant_id bigint not null,
        bachelors varchar(80) not null,
        degree varchar(80) not null,
        grad_gpa real not null,
        research_area varchar(100) not null,
        office_day varchar(9) null default 'none',
        office_hour_start time null,
        office_hour_end time null,
        unique (assistant_id)
    );

.. code-block:: sql
    :linenos:
    :caption: SQL of Student Table

    CREATE domain credit as real check (
        ((value >= 0) and (value <=250))
    );

    CREATE TABLE IF NOT EXISTS STUDENT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id)  on delete cascade on update cascade NOT NULL,
        faculty_id int references FACULTY(id) on delete cascade on update cascade not null,
        department_id int references DEPARTMENT(id) on delete cascade on update cascade not null,
        student_id bigint not null,
        semester smallint not null default 1,
        grade smallint not null default 1,
        gpa real not null default 0,
        credits_taken credit not null,
        minor boolean not null default false,
        unique (student_id)
    );

Inserting Data
++++++++++++++

Filling tables of ITU DataBees with initialization data for easier demonstration of the functions.

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into People Table

    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (1,'fadmin', 'fatih', '1', 
    'fadmin@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (2,'badmin', 'batu', '1', 
    'badmin@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (3,'cadmin', 'cihat', '1', 
    'cadmin@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (4,'zadmin', 'zeynep', '1', 
    'zadmin@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11,'finstructor', 'fatih', '1', 
    'finstructor@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22,'binstructor', 'batu', '1', 
    'binstructor@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33,'cinstructor', 'cihat', '1', 
    'cinstructor@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44,'zinstructor', 'zeynep', '1', 
    'zinstructor@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (111,'fstudent', 'fatih', '1', 
    'fstudent@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (222,'bstudent', 'batu', '1', 
    'bstudent@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (333,'cstudent', 'cihat', '1', 
    'cstudent@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (444,'zstudent', 'zeynep', '1', 
    'zstudent@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11111,'fassistant', 'fatih', '1', 
    'fassistant@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22222,'bassistant', 'batu', '1', 
    'bassistant@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33333,'cassistant', 'cihat', '1', 
    'cassistant@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44444,'zassistant', 'zeynep', '1', 
    'zassistant@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '3', '2019-10-10', 'a','b');

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into Assistant Table

    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (11111, 1, 11, 500180707, 'ITU', 'Computer Engineering', 3.7, 'Machine Learning', 'Monday', '10:30', '12:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (22222, 2, 22, 500180704, 'ITU', 'Computer Engineering', 3.8, 'Machine Learning', 'Tuesday', '12:30', '14:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (33333, 3, 33, 500180705, 'ITU', 'Computer Engineering', 3.9, 'Machine Learning', 'Wednesday', '14:30', '16:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (44444, 4, 44, 500150150, 'ITU', 'Computer Engineering', 4.0, 'Machine Learning', 'Thursday', '16:30', '18:30');

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into Student Table

    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (111, 1, 1, 150180707, 69.5, 0);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (222, 1, 2, 150180704, 200, 4);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (333, 2, 1, 150180705, 200, 4);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (444, 3, 1, 150150150, 200, 4);

Re-Initialization of DataBees
+++++++++++++++++++++++++++++

Code below removes clears all content of the database.

.. code-block:: sql
    :linenos:
    :caption: SQL for Resetting the Database

    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;

Classes
-------

Classes are python objects for the user types that ITU DataBees uses.

Classes implemented by this user can be examined in detailed with the given source code from various files.

Classes person, assistant and student are created and implemented by this user and their corresponding codes are given below.

.. code-block:: python
    :linenos:
    :caption: Person class from ``person.py``

    class Person(UserMixin):
        def __init__(self, tr_id, name, surname, phone_number, email, password, person_category,
                     mother_fname, father_fname, gender, birth_city, birth_date, id_reg_city,
                     id_reg_district, photo_name, photo_extension, photo_data):
            self.tr_id = tr_id
            self.id = email
            self.name = name
            self.surname = surname
            self.phone_number = phone_number
            self.email = email
            self.password = password
            self.person_category = person_category
            self.mother_fname = mother_fname
            self.father_fname = father_fname
            self.gender = gender
            self.birth_city = birth_city
            self.birth_date = birth_date
            self.id_reg_city = id_reg_city
            self.id_reg_district = id_reg_district
            self.photo_name = photo_name
            self.photo_extension = photo_extension
            self.photo_data = photo_data

            if self.photo_data:
                self.photo = b64encode(self.photo_data)
                self.photo = self.photo.decode('utf-8')

            self.active = True
            if self.person_category == 0:
                self.is_admin = True
            else:
                self.is_admin = False

            @property
            def get_id(self):
                return self.email

            @property
            def is_active(self):
                return self.active

.. code-block:: python
    :linenos:
    :caption: Assistant class from ``assistant.py``

    class Assistant(UserMixin):
    def __init__(self, tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa,
                 research_area, office_day, office_hour_start, office_hour_end):
        self.tr_id = tr_id
        self.faculty_id = faculty_id
        self.supervisor = supervisor
        self.assistant_id = assistant_id
        self.bachelors = bachelors
        self.degree = degree
        self.grad_gpa = grad_gpa
        self.research_area = research_area
        self.office_day = office_day
        self.office_hour_start = office_hour_start
        self.office_hour_end = office_hour_end

.. code-block:: python
    :linenos:
    :caption: Student class from ``student.py``

    class Student(UserMixin):
        def __init__(self, tr_id, faculty_id, department_id, student_id, semester, grade, gpa,
                     credits_taken, minor):
            self.tr_id = tr_id
            self.faculty_id = faculty_id
            self.department_id = department_id
            self.student_id = student_id
            self.semester = semester
            self.grade = grade
            self.gpa = gpa
            self.credits_taken = credits_taken
            self.minor = minor

