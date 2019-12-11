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
        unique (tr_id, email)
    );
    
    CREATE domain credit as real check (
        ((value >= 15) and (value <=28))
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
        name 				VARCHAR(40)	NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        address 			VARCHAR(40),
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id)
        FOREIGN KEY(campus_id) REFERENCES CAMPUS(id)
    );

    CREATE TABLE IF NOT EXISTS DEPARTMENT(
        id				    SERIAL 		NOT NULL,
        faculty_id			INT			NOT NULL,
        name 				VARCHAR(40)	NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        block_number 		CHAR(1),
        budget			 	INT,
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(faculty_id) REFERENCES FACULTY(id)
    );

    CREATE TABLE IF NOT EXISTS CLASSROOM(
        classroom_id INT NOT NULL PRIMARY KEY,
        capacity INT NOT NULL,
        has_projection BOOLEAN NOT NULL,
        door_number CHAR(4) NOT NULL,
        floor VARCHAR(2) NOT NULL,
        renewed BOOLEAN DEFAULT false,
        board_count CHAR(1),
        air_conditioner BOOLEAN DEFAULT false,
        faculty_id INT,
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id)
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
        tr_id BIGINT NOT NULL PRIMARY KEY,
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
        supervisor bigint references INSTRUCTOR(tr_id) not null,
        assistant_id bigint not null,
        bachelors varchar(80) not null,
        degree varchar(80) not null,
        grad_gpa real not null,
        research_area varchar(100) not null,
        office_day varchar(9) null default 'None',
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
        FOREIGN KEY (classroom_id) REFERENCES CLASSROOM (classroom_id),
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id),
        FOREIGN KEY (instructor_id) REFERENCES INSTRUCTOR (tr_id)
    );
    
    CREATE TABLE IF NOT EXISTS COURSE_ASSISTED (
        crn char(6) primary key references COURSE(crn) not null,
        assistant_id bigint references ASSISTANT(assistant_id) not null,
        room_id int references CLASSROOM(classroom_id) not null,
        problem_session boolean not null default false,
        exam boolean not null default false,
        homework boolean not null default false,
        quiz boolean not null default false,
        recitation boolean not null default false,
        role varchar(60) null
    );

    CREATE TABLE IF NOT EXISTS TAKEN_COURSES(
        id INT PRIMARY KEY,
        student_id BIGINT NOT NULL,
        crn CHAR(6) NOT NULL,
        datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES STUDENT (tr_id),
        FOREIGN KEY (crn) REFERENCES COURSE (crn) 
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
    
    """
]


def initialize(url):
    with dbapi2.connect(url) as connection:
        cursor = connection.cursor()
        for statement in CLEAR_SCHEMA:
            cursor.execute(statement)
        for statement in INIT_STATEMENTS:
            cursor.execute(statement)
        cursor.close()


if __name__ == "__main__":
    url = os.getenv("DATABASE_URL")
    if url is None:
        print("Usage: DATABASE_URL=url python dbinit.py", file=sys.stderr)
        sys.exit(1)
    initialize(url)
    print("Successfully initialized the DataBees!")
