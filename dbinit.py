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