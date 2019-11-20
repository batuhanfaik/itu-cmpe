import os

import psycopg2 as dbapi2
from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

import campus_views
import views
from database import Database
from person import Person

UPLOAD_FOLDER = '/media'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
SECRET_KEY = os.urandom(32)
lm = LoginManager()

lm.login_view = "views.login"


def init_db(db_url):
    db = Database(db_url)
    try:  # TODO: Fix the UniqueViolation error even though the query is unique
        db.add_person(
            Person("11111111110", "Ahmet", "Mehmet", "+905508004060", "mehmet19@itu.edu.tr",
                   "password", 2, "Fatma", "Ali", "M", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
        db.add_person(
            Person("12111111110", "Veli", "Mehmet", "+905508004060", "veli19@itu.edu.tr",
                   "password", 0, "Fatma", "Ali", "M", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
        db.add_person(
            Person("13111111110", "Deli", "Mehmet", "+905508004060", "deli19@itu.edu.tr",
                   "password", 5, "Fatma", "Ali", "M", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
        db.add_person(
            Person("14111111110", "Ucar", "Mehmet", "+905508004060", "ucar19@itu.edu.tr",
                   "password", 3, "Fatma", "Ali", "F", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
        db.add_person(
            Person("15111111110", "Ucmaz", "Mehmet", "+905508004060", "ucmaz19@itu.edu.tr",
                   "password", 5, "Fatma", "Ali", "M", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
        db.add_person(
            Person("16111111110", "Ucarmi", "Mehmet", "+905508004060", "ucarmi19@itu.edu.tr",
                   "password", 1, "Fatma", "Ali", "F", "Istanbul", "01-01-2000", "Istanbul",
                   "Sariyer"))
    except dbapi2.errors.UniqueViolation:
        print("A person with the same TR ID already exists")
    return db


def create_app(db_url):
    app = Flask(__name__, static_url_path="/static")
    csrf.init_app(app)
    # Triangle(app)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config.from_object("settings")
    app.config['SECRET_KEY'] = SECRET_KEY

    app.add_url_rule("/", view_func=views.landing_page, methods=['GET', 'POST'])
    app.add_url_rule("/login", view_func=views.login_page, methods=['GET', 'POST'])
    app.add_url_rule("/logout", view_func=views.logout_page, methods=['GET', 'POST'])
    app.add_url_rule("/people", view_func=views.people_page, methods=['GET', 'POST'])
    app.add_url_rule("/people/<tr_id>", view_func=views.person_page, methods=['GET', 'POST'])
    app.add_url_rule("/campuses/campus", view_func=campus_views.campus, methods=['GET', 'POST'])
    db = init_db(db_url)
    app.config["db"] = db
    lm.init_app(app)
    lm.login_view = "login_page"
    return app


csrf = CSRFProtect()
db_url = os.getenv("DATABASE_URL")
app = create_app(db_url)

if __name__ == "__main__":
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    debug = app.config.get("DEBUG")
    app.run(host=host, port=port, debug=debug)
