import os
import sys

import psycopg2 as dbapi2

CLEAR_SCHEMA = [
    """
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;
    """
]

INIT_STATEMENTS = [
    """    
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
        id_reg_city varchar(50) not null,
        id_reg_district varchar(50) not null,
        photo_name varchar(256),
        photo_extension varchar(10),
        photo_data bytea, 
        unique (tr_id, email)
    );
    
    CREATE domain credit as real check (
        ((value >= 0) and (value <=250))
    );
        
    CREATE TABLE IF NOT EXISTS CAMPUS(
        id		            SERIAL 		NOT NULL,
        name 		        VARCHAR(25)	NOT NULL,
        address 	        VARCHAR(40)	NOT NULL,
        city 		        VARCHAR(25),
        size 		        INT,
        foundation_date     DATE,
        phone_number        VARCHAR(12),   
        campus_image_name   VARCHAR(400),
        campus_image_extension VARCHAR(10),
        campus_image_data   bytea, 
        PRIMARY KEY(id)
    );
    
    CREATE TABLE IF NOT EXISTS FACULTY(
        id				    SERIAL 		NOT NULL,
        campus_id           INT         NOT NULL,
        name 				VARCHAR(100) NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        address 			VARCHAR(40),
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS(id)
    );

    CREATE TABLE IF NOT EXISTS DEPARTMENT(
        id				    SERIAL 		NOT NULL,
        faculty_id			INT			NOT NULL,
        name 				VARCHAR(100) NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        block_number 		CHAR(1),
        budget			 	INT,
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(faculty_id) REFERENCES FACULTY(id)
    );

    CREATE TABLE IF NOT EXISTS CLASSROOM(
        id              SERIAL      NOT NULL PRIMARY KEY,
        capacity        INT         NOT NULL,
        has_projection  BOOLEAN     DEFAULT false,
        door_number     CHAR(4)     NOT NULL,
        floor           VARCHAR(2),
        renewed         BOOLEAN     DEFAULT false,
        board_count     CHAR(1),
        air_conditioner BOOLEAN     DEFAULT false,
        faculty_id      INT         NOT NULL,
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id),
        unique(door_number, faculty_id)
    );


    CREATE TABLE IF NOT EXISTS STUDENT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id) NOT NULL,
        faculty_id int references FACULTY(id) not null,
        department_id int references DEPARTMENT(id) not null,
        student_id bigint not null,
        semester smallint not null default 1,
        grade smallint not null default 1,
        gpa real not null default 0,
        credits_taken credit not null,
        minor boolean not null default false,
        unique (student_id)
    );

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
        FOREIGN KEY (tr_id) REFERENCES PEOPLE (tr_id),
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id),
        FOREIGN KEY (department_id) REFERENCES DEPARTMENT (id)
    );
    
    CREATE TABLE IF NOT EXISTS ASSISTANT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id) NOT NULL,
        faculty_id int references FACULTY(id) not null,
        supervisor bigint references INSTRUCTOR(id) not null,
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
    
    CREATE TABLE IF NOT EXISTS COURSE (
        crn CHAR(6) NOT NULL PRIMARY KEY,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        day VARCHAR(9) NOT NULL,
        capacity INT NOT NULL,
        enrolled INT NOT NULL,
        credits REAL NOT NULL,
        language CHAR(2) NOT NULL,
        classroom_id INT NOT NULL,
        faculty_id INT NOT NULL,
        instructor_id BIGINT NOT NULL,
        FOREIGN KEY (classroom_id) REFERENCES CLASSROOM (id),
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id),
        FOREIGN KEY (instructor_id) REFERENCES INSTRUCTOR (id)
    );
    
    CREATE TABLE IF NOT EXISTS COURSE_ASSISTED (
        crn char(6) primary key references COURSE(crn) not null,
        assistant_id bigint references ASSISTANT(assistant_id) not null,
        room_id int references CLASSROOM(id) not null,
        problem_session boolean not null default false,
        exam boolean not null default false,
        homework boolean not null default false,
        quiz boolean not null default false,
        recitation boolean not null default false,
        role varchar(60) null
    );

    CREATE TABLE IF NOT EXISTS TAKEN_COURSE(
        id SERIAL PRIMARY KEY,
        student_id BIGINT NOT NULL,
        crn CHAR(6) NOT NULL,
        datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES STUDENT (student_id),
        FOREIGN KEY (crn) REFERENCES COURSE (crn), 
        UNIQUE(student_id, crn)
    );
    
    CREATE TABLE IF NOT EXISTS COMPETED_COURSE(
        id                  SERIAL      NOT NULL PRIMARY KEY,
        student_id          BIGINT      NOT NULL,
        crn                 CHAR(6)     NOT NULL,
        grade               CHAR(2)     NOT NULL,
        FOREIGN KEY (student_id) REFERENCES STUDENT (student_id),
        FOREIGN KEY (crn) REFERENCES COURSE (crn),
        UNIQUE(student_id, crn)
    );

    CREATE TABLE IF NOT EXISTS ADMINISTRATOR(
        tr_id           BIGINT          NOT NULL,
        faculty_id 	    INT             NOT NULL, 
        phone_number 	VARCHAR(40)	    NOT NULL,
        FOREIGN KEY(tr_id) REFERENCES PEOPLE (tr_id)
    );
    
    CREATE TABLE IF NOT EXISTS FACILITY(
        id				    BIGINT 		NOT NULL,
        campus_id           SERIAL      NOT NULL,
        name 				VARCHAR(40)	NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        number_of_workers	INT,
        size             	INT         NOT NULL,
        expenses    		INT,
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS (id)
    );
    
    CREATE TABLE IF NOT EXISTS STAFF(
        id              BIGINT          NOT NULL,
        manager_name    VARCHAR(20)     NOT NULL, 
        absences	    INT             NOT NULL, 
        hire_date      	DATE            NOT NULL,
        authority_lvl   INT             NOT NULL,
        department      VARCHAR(20)     NOT NULL,
        social_sec_no   BIGINT          NOT NULL,
        PRIMARY KEY(id),
        FOREIGN KEY(id) REFERENCES PEOPLE (tr_id)
    );
    
    CREATE TABLE IF NOT EXISTS STAFF_FACIL(
        title           VARCHAR(20)     NOT NULL,
        from_date 	    DATE            NOT NULL, 
        to_date 	    DATE, 
        salary  	    INT             NOT NULL, 
        facility_id	    BIGINT          NOT NULL, 
        staff_id        BIGINT          NOT NULL,
        duty         	VARCHAR(20)	    NOT NULL,
        FOREIGN KEY(facility_id) REFERENCES FACILITY (id),
        FOREIGN KEY(staff_id) REFERENCES STAFF (id),
        PRIMARY KEY(facility_id,staff_id)
    );
        
    """,
    # DATABASE FILLER #
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (1,'fadmin', 'fatih', '1', 
    'fadmin@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '0', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (2,'badmin', 'batu', '1', 
    'badmin@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '0', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (3,'cadmin', 'cihat', '1', 
    'cadmin@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '0', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (4,'zadmin', 'zeynep', '1', 
    'zadmin@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '0', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11,'finstructor', 'fatih', '1', 
    'finstructor@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '2', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22,'binstructor', 'batu', '1', 
    'binstructor@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '2', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33,'cinstructor', 'cihat', '1', 
    'cinstructor@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '2', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44,'zinstructor', 'zeynep', '1', 
    'zinstructor@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '2', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (111,'fstudent', 'fatih', '1', 
    'fstudent@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '5', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (222,'bstudent', 'batu', '1', 
    'bstudent@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '5', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (333,'cstudent', 'cihat', '1', 
    'cstudent@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '5', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (444,'zstudent', 'zeynep', '1', 
    'zstudent@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '5', '2019-10-10', 'a','b');""",

    # Create campus
    """insert into campus (name, address) values ('Ayazaga', 'ayazaga iste aq');""",

    # Add some faculties
    """insert into faculty (campus_id, name, shortened_name) values (1, 
        'Faculty of Computer and Informatics Engineering ', 'CMPF');""",
    """insert into faculty (campus_id, name, shortened_name) values (1, 
    'Faculty of Electric and Electronics Engineering', 'EEB');""",
    """insert into faculty (campus_id, name, shortened_name) values (1, 'Faculty of ISLETME YEAH', 'ISLF');""",
    
    # Add departments
    """insert into department (faculty_id, name, shortened_name) values (1, 'Computer Engineering', 'BLG');""",
    """insert into department (faculty_id, name, shortened_name) values (1, 'Informatics Engineering', 'BIL');""",
    """insert into department (faculty_id, name, shortened_name) values (2, 
    'Electronic and Communication Engineering', 'EHB');""",
    """insert into department (faculty_id, name, shortened_name) values (2, 
    'Elektronik Haberlesme Turkce', 'EHBTR');""",
    """insert into department (faculty_id, name, shortened_name) values (3, 
    'Isletme Muhendisligi', 'ISLTR');""",
    """insert into department (faculty_id, name, shortened_name) values (3, 
    'Management Enginnering', 'ISL');""",


    # Add classrooms


]


def reset_db(url):
    with dbapi2.connect(url) as connection:
        cursor = connection.cursor()
        for statement in CLEAR_SCHEMA:
            cursor.execute(statement)
        for statement in INIT_STATEMENTS:
            try:
                cursor.execute(statement)
            except dbapi2.Error as e:
                print(e)
                sys.exit(1)
        cursor.close()


if __name__ == "__main__":
    url = os.getenv("DATABASE_URL")
    if url is None:
        print("Usage: DATABASE_URL=url python dbinit.py", file=sys.stderr)
        sys.exit(1)
    reset_db(url)
    print("Successfully initialized the DataBees!")
