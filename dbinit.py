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
        name 		        VARCHAR(50)	NOT NULL,
        address 	        VARCHAR(80)	NOT NULL,
        city 		        VARCHAR(25),
        size 		        INT,
        foundation_date     DATE,
        phone_number        CHAR(11),   
        campus_image_extension VARCHAR(10) DEFAULT('NO_IMAGE'),
        campus_image_data   bytea, 
        PRIMARY KEY(id)
    );
    CREATE TABLE IF NOT EXISTS FACULTY(
        id				    SERIAL 		NOT NULL,
        campus_id           INT         NOT NULL,
        name 				VARCHAR(100) NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        address 			VARCHAR(80),
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS(id) on delete cascade on update cascade
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
        FOREIGN KEY(faculty_id) REFERENCES FACULTY(id) on delete cascade on update cascade
    );
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
    CREATE TABLE IF NOT EXISTS STUDENT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id)  on delete cascade on update cascade NOT NULL,
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
        FOREIGN KEY (tr_id) REFERENCES PEOPLE (tr_id)  on delete cascade on update cascade,
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id) on delete cascade on update cascade,
        FOREIGN KEY (department_id) REFERENCES DEPARTMENT (id) on delete cascade on update cascade,
        unique(tr_id)
    );
    CREATE TABLE IF NOT EXISTS ASSISTANT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id) on delete cascade on update cascade NOT NULL,
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
    CREATE TABLE IF NOT EXISTS SYLLABUS (
        crn             char(5)         PRIMARY KEY,
        file            bytea           default null,
        foreign key (crn) references course(crn) on delete cascade on update cascade
    );
    CREATE TABLE IF NOT EXISTS TAKEN_COURSE(
        id SERIAL PRIMARY KEY,
        student_id BIGINT NOT NULL,
        crn CHAR(6) NOT NULL,
        grade REAL NULL,
        datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES STUDENT (student_id)  on delete cascade on update cascade,
        FOREIGN KEY (crn) REFERENCES COURSE (crn)  on delete cascade on update cascade, 
        UNIQUE(student_id, crn),
        CHECK ( grade >= 0 and grade <= 4 ) 
    );
    CREATE TABLE IF NOT EXISTS FACILITY(
        id				    SERIAL 		NOT NULL,
        campus_id           SERIAL      NOT NULL,
        name 				VARCHAR(40)	NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        number_of_workers	INT,
        size             	INT,
        expenses    		INT,
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS (id) on delete cascade on update cascade
    );
    CREATE TABLE IF NOT EXISTS STAFF(
        id              BIGINT not null,
        manager_name    VARCHAR(40) null, 
        absences	    INT null, 
        hire_date      	DATE null,
        authority_lvl   INT null,
        department      VARCHAR(40) null,
        social_sec_no   INT null,
        PRIMARY KEY(id),
        FOREIGN KEY(id) REFERENCES PEOPLE (tr_id) on delete cascade on update cascade
    );
    CREATE TABLE IF NOT EXISTS STAFF_FACIL(
        title           VARCHAR(20)     NOT NULL,
        from_date 	    DATE            NOT NULL, 
        to_date 	    DATE, 
        salary  	    INT             NOT NULL, 
        facility_id	    BIGINT          NOT NULL, 
        staff_id        BIGINT          NOT NULL,
        duty         	VARCHAR(20)	    NOT NULL,
        FOREIGN KEY(facility_id) REFERENCES FACILITY (id) on delete cascade on update cascade,
        FOREIGN KEY(staff_id) REFERENCES STAFF (id) on delete cascade on update cascade, 
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
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11111,'fassistant', 'fatih', '1', 
    'fassistant@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '3', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22222,'bassistant', 'batu', '1', 
    'bassistant@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '3', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33333,'cassistant', 'cihat', '1', 
    'cassistant@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '3', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44444,'zassistant', 'zeynep', '1', 
    'zassistant@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '3', '2019-10-10', 'a','b');""",

    # Create campus
    """insert into campus (name, address,city,size,foundation_date,phone_number) values ('Ayazağa', 'Reşitpaşa, Park Yolu No:2, 34467 Sarıyer','İstanbul','247','01.01.1773','2122853030');""",
    """insert into campus (name, address,city,size,foundation_date,phone_number) values ('Taşkışla', 'Harbiye Mh, Taşkışla Cd. No:2, 34367 Şişli','İstanbul','52','01.01.1950','2122931300');""",
    """insert into campus (name, address,city,size,foundation_date,phone_number) values ('Maçka', 'Harbiye, İTÜ Maçka Kampüsü 4 A, 34367 Şişli','İstanbul','63','01.01.1970','2122963147');""",
    """insert into campus (name, address,city,size,foundation_date,phone_number) values ('Gümüşsuyu', 'Gümüşsuyu, İnönü Cd. No:65, 34437 Beyoğlu','İstanbul','42','01.01.1850','2122931300');""",
    """insert into campus (name, address,city,size,foundation_date,phone_number) values ('Tuzla', 'Postane, Denizcilik Fakültesi, 34940 Tuzla','İstanbul','66','01.01.1992','2163954501');""",

    # Add some faculties
    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (1, 
        'Computer and Informatics Engineering', 'CMPF','İTÜ Ayazağa Kampüsü Bilgisayar ve Bilişim Fakültesi 34467 Sariyer','01.01.2010','2122853682');""",
    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (1, 
        'Electrical and Electronics Engineering ', 'EEBF','Reşitpaşa, Prof. B. Karafakıoğlu Cd, 34467 Sarıyer','01.01.1934','2122853422');""",
    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (1, 
        'Science and Letters', 'SCF','Reşitpaşa, 34469 Maslak/Sarıyer/İstanbul 34467 Sariyer','01.01.1971','2122853340');""",
    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (1, 
        'Civil Engineering', 'CEF','Ayazağa Yerleşkesi İnşaat Fakültesi Binası, Maslak, 34469 Sarıyer/İstanbul','01.01.1727','2122853855');""",

    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (4, 
        'Mechanical Engineering', 'MEF','Gümüşsuyu, İnönü Cd. No:65, 34437 Beyoğlu/İstanbul','01.01.1944','2122931300');""",
    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (4, 
        'Textile Technologies and Design', 'TTDF','Gümüşsuyu, İnönü Cd. No:65, 34437 Beyoğlu/İstanbul','01.01.2004','2122931300');""",

    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (2, 
        'Architecture', 'AF','Harbiye Mh, Taşkışla Cd. No:2, 34367 Şişli/İstanbul','01.01.1884','2122931300');""",

    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (3, 
        'Management', 'MF','Harbiye, 34367 Maçka/Beşiktaş/İstanbul','01.01.1977','2122931300');""",

    """insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (5, 
        'Maritime', 'FOM','Postane Mahallesi Manastır Yolu Caddesi 1 1, 34940 Tuzla','01.01.1884','2163954501');""",


    # Add departments

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (1, 
    'Computer Engineering', 'BLG','D','897987','01.01.1997','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (2, 
    'Electrical Engineering', 'BLG','D','879797','01.01.1970','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (2, 
    'Electronics & Communication Engineering', 'BLG','D','987987','01.01.1978','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (3, 
    'Mathematics', 'MAT','A','76876','01.01.1850','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (3, 
    'Physics Engineering', 'PHI','B','78676','01.01.1820','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (3, 
    'Chemistry', 'CHE','C','987987','01.01.1810','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (4, 
    'Civil Engineering', 'CIV','A','788678','01.01.1950','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (4, 
    'Geomatics Engineering', 'GEO','B','34245','01.01.1950','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (4, 
    'Enviromental Engineering', 'ENV','C','543453','01.01.1930','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (5, 
    'Mechanical Engineering', 'MEC','B','897897','01.01.2001','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (5, 
    'Manufacturing Engineering', 'MAK','A','876878','01.01.2003','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (6, 
    'Textile Engineering', 'TEX','A','654564','01.01.1970','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (6, 
    'Fashion Design', 'FASH','D','8787678','01.01.1978','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (6, 
    'Textile Development and Design', 'TEXD','C','989799','01.01.1987','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (7, 
    'Architecture', 'ARC','D','897979','01.01.1790','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (7, 
    'Urban and Regional Planning', 'URP','E','876876','01.01.1890','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (7, 
    'Industrial Design', 'IND','A','567475','01.01.1990','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (7, 
    'Landscape Architecture', 'LAND','B','900000','01.01.1990','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (8, 
    'Management Engineering', 'MAN','C','80000','01.01.1897','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (8, 
    'Industrial Engineering', 'IND','B','300000','01.01.1890','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (8, 
    'Economics', 'EKO','A','1000000','01.01.1893','2122853682');""",

    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (9, 
    'Marine Engineering', 'BLG','B','300500','01.01.1990','2122853682');""",
    """insert into department (faculty_id, name, shortened_name,block_number,budget,foundation_date,phone_number) values (9, 
    'Maritime Transportation and Management Engineering', 'MTME','A','500000','01.01.1997','2122853682');""",

    # Add classrooms
    """insert into classroom (capacity, door_number, faculty_id) values ('100', '5202', '1');""",
    """insert into classroom (capacity, door_number, faculty_id) values ('120', '5204', '1');""",
    """insert into classroom (capacity, door_number, faculty_id) values ('31', 'A101', '2');""",
    """insert into classroom (capacity, door_number, faculty_id) values ('62', 'A102', '2');""",
    """insert into classroom (capacity, door_number, faculty_id) values ('104', 'DB11', '3');""",
    """insert into classroom (capacity, door_number, faculty_id) values ('300', 'DB12', '3');""",

    # Add students
    """insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (111, 1, 1, 150180707, 69.5, 0);""",
    """insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (222, 1, 2, 150180704, 200, 4);""",
    """insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (333, 2, 1, 150180705, 200, 4);""",
    """insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (444, 3, 1, 150150150, 200, 4);""",

    # Add instructor
    """insert into instructor (tr_id, department_id, faculty_id) values (11, 1, 1);""",
    """insert into instructor (tr_id, department_id, faculty_id) values (22, 2, 1);""",
    """insert into instructor (tr_id, department_id, faculty_id) values (33, 1, 2);""",
    """insert into instructor (tr_id, department_id, faculty_id) values (44, 1, 3);""",

    # Add Courses
    """insert into course (crn, code, name, start_time, end_time, day, capacity, credits, classroom_id,
      instructor_id, department_id) values ('11111', '101', 'Intro to computing (C)', '12:00:00', '13:00:00',
     'Wednesday', 50, 3, 1, 1, 1);""",
    """insert into course (crn, code, name, start_time, end_time, day, capacity, credits, classroom_id,
     instructor_id, department_id) values ('22222', '102', 'Intro to computing (Python)', '12:00:00', '13:00:00',
     'Thursday', 50, 3, 1, 1, 1);""",

    # Add Staff
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('1', 'Manager1', '1', '2019-12-12','12345','Finance ','1');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('2', 'Manager2', '0', '2019-12-12','12344','Information Tech','2');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('3', 'Manager3', '1', '2019-12-12','12345','Information Tech','1');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('4', 'Manager4', '0', '2019-12-12','12344','Service','2');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('11', 'Manager1', '1', '2019-12-12','12345','Finance','1');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('22', 'Manager2', '0', '2019-12-12','12344','Information Tech','2');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('33', 'Manager3', '1', '2019-12-12','12345','Information Tech','1');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('44', 'Manager4', '0', '2019-12-12','12344','Service Tech','2');""",
    """insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('111', null, null , null,null,null,null);""",

    # Insert Taken Courses
    """insert into taken_course (id,student_id,crn) values ('1','150180704','11111');""",
    """insert into taken_course (id,student_id,crn) values ('2','150180705','11111');""",
    """insert into taken_course (id,student_id,crn) values ('3','150180707','22222');""",
    """insert into taken_course (id,student_id,crn) values ('4','150150150','22222');""",

    # Add facility
    """insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (1, 1, 'Yemekhane', 'YMK', '50', '1400', '70000')""",
    """insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (2, 2, 'Kütüphane', 'LIB', '50', '1400', '50000')""",
    """insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (3, 4, 'Bilgi İşlem', 'BIDB', '50', '1400', '80000')""",
    # Add Staff-facility connection
    """insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('leader', '2019-12-12', '2019-12-12', '2000', 1, 1, 'something')""",
    """insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('security','2019-12-12', '2019-12-12', '2000', 2, 2, 'leader')""",
    """insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('member', '2019-12-12', '2019-12-12', '2000', 2, 3, 'member')""",

    # Staff people
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
        birth_date, id_reg_city, id_reg_district) values (1111,'fstaff', 'fatih', '1', 
        'fstaff@itu.edu.tr','fatih',
        '1', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
        birth_date, id_reg_city, id_reg_district) values (2222,'bstaff', 'batu', '1', 
        'bstaff@itu.edu.tr','batu',
        '1', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
        birth_date, id_reg_city, id_reg_district) values (3333,'cstaff', 'cihat', '1', 
        'cstaff@itu.edu.tr','cihat',
        '1', '2019-10-10', 'a','b');""",
    """insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
        birth_date, id_reg_city, id_reg_district) values (4444,'zstaff', 'zeynep', '1', 
        'zstaff@itu.edu.tr','zeynep',
        '1', '2019-10-10', 'a','b');""",


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
