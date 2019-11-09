import psycopg2 as dbapi2

INIT_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS PEOPLE (
        tr_id BIGINT PRIMARY KEY NOT NULL,
        name VARCHAR(40) NOT NULL,
        surname VARCHAR(40) NOT NULL,
        phone_number varchar(20) not null,
        email VARCHAR(60) NOT NULL,
        pass varchar(256) not null default tr_id,
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
    
    CREATE TABLE IF NOT EXISTS COURSES_ASSISTED (
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
        FOREIGN KEY (classroom_id) REFERENCES CLASSROOM (classroom_id),
        FOREIGN KEY (faculty_id) REFERENCES DEPARTMENT (id)
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
        FOREIGN KEY (classroom_id) REFERENCES CLASSROOM (classroom_id),
        FOREIGN KEY (faculty_id) REFERENCES DEPARTMENT (id)
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
        FOREIGN KEY (faculty_id) REFERENCES FACULTY (id)
    );

    CREATE TABLE IF NOT EXISTS TAKEN_COURSES(
        id INT PRIMARY KEY,
        student_id BIGINT NOT NULL,
        crn CHAR(6) NOT NULL,
        datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES STUDENT (tr_id),
        FOREIGN KEY (crn) REFERENCES COURSE (crn) 
    );
    """

]


def initialize(url):
    with dbapi2.connect(url) as connection:
        cursor = connection.cursor()
        for statement in INIT_STATEMENTS:
            cursor.execute(statement)
        cursor.close()


if __name__ == "__main__":
    url = "postgres://aibqztyjqfdboa:07aad1d3462c03868d5c069697e882eb39e24cfa8ebd35c6829421117ba66325@ec2-54-246-100-246.eu-west-1.compute.amazonaws.com:5432/d80l4n8sl73rk0"
    initialize(url)