import psycopg2 as dbapi2

INIT_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS PEOPLE (
        tr_id BIGINT PRIMARY KEY NOT NULL,
        name VARCHAR(40) NOT NULL,
        surname VARCHAR(40) NOT NULL,
        email VARCHAR(60) NOT NULL,
        person_category SMALLINT NOT NULL
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
        FOREIGN KEY classroom_id REFERENCES CLASSROOM (classroom_id),
        FOREIGN KEY faculty_id REFERENCES DEPARTMENT (id),
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
        FOREIGN KEY tr_id REFERENCES PEOPLE (tr_id),
        FOREIGN KEY classroom_id REFERENCES CLASSROOM (classroom_id),
        FOREIGN KEY faculty_id REFERENCES DEPARTMENT (id),
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
        FOREIGN KEY faculty_id REFERENCES FACULTY (id),
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