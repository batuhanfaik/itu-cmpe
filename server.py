import os

import psycopg2 as dbapi2
from flask import Flask

import views
from database import Database
from person import Person


def init_db(db_url):
    db = Database(db_url)
    try:  # TODO: Fix the UniqueViolation error even though the query is unique
        db.add_person(
            Person("11111111110", "Ahmet", "Mehmet", "+905508004060", "mehmet19@itu.edu.tr",
                   "password", 5, "Fatma", "Ali", "E", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
    except dbapi2.errors.UniqueViolation:
        print("A person with the same TR ID already exists")
    return db


def create_app(db_url):
    app = Flask(__name__, static_url_path="/static")

    app.config.from_object("settings")

    app.add_url_rule("/", view_func=views.landing_page, methods=['GET', 'POST'])
    app.add_url_rule("/login", view_func=views.login_page, methods=['GET', 'POST'])
    app.add_url_rule("/people", view_func=views.people_page, methods=['GET', 'POST'])
    app.add_url_rule("/people/<tr_id>", view_func=views.person_page, methods=['GET', 'POST'])
    app.add_url_rule("/campuses", view_func=views.manage_campuses,
                     methods=['GET', 'POST', "DELETE"])
    db = init_db(db_url)
    app.config["db"] = db
    return app


db_url = os.getenv("DATABASE_URL")
app = create_app(db_url)

if __name__ == "__main__":
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    debug = app.config.get("DEBUG")
    app.run(host=host, port=port, debug=debug)
