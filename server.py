import os

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

import campus_views
import views
from database import Database

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'media'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'svg'}
SECRET_KEY = "secret"
lm = LoginManager()
csrf = CSRFProtect()
lm.login_view = "views.login_page"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db(db_url):
    db = Database(db_url)
    return db


def create_app(db_url):
    app = Flask(__name__, static_url_path="/static")
    # Triangle(app)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['APP_ROOT'] = APP_ROOT
    app.config.from_object("settings")
    csrf.init_app(app)

    app.add_url_rule("/", view_func=views.landing_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/login", view_func=views.login_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/logout", view_func=views.logout_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/people", view_func=views.people_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/people/<tr_id>",
                     view_func=views.person_page, methods=['GET', 'POST'])
    app.add_url_rule("/students", view_func=views.students_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/students/<tr_id>",
                     view_func=views.student_page, methods=['GET', 'POST'])
    app.add_url_rule("/reset_db", view_func=views.reset_db, methods=["POST"])
    app.add_url_rule("/assistants", view_func=views.assistants_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/assistants/<tr_id>",
                     view_func=views.assistant_page, methods=['GET', 'POST'])
    app.add_url_rule("/campuses/campus",
                     view_func=campus_views.campus, methods=['GET', 'POST'])
    app.add_url_rule("/campuses/<campus_id>",
                     view_func=campus_views.campus_detailed, methods=['GET', 'POST'])
    app.add_url_rule("/faculty/<faculty_id>",
                     view_func=campus_views.faculty_detailed, methods=['GET', 'POST'])
    app.add_url_rule("/faculty/<faculty_id>/add_classroom",
                     view_func=views.add_classroom_page, methods=['GET', 'POST'])
    app.add_url_rule("/faculty/<faculty_id>/classroom/<id>/edit", view_func=views.edit_classroom_page,
                     methods=['GET', 'POST'])
    app.add_url_rule("/department/<department_id>",
                     view_func=campus_views.department_detailed, methods=['GET', 'POST'])
    app.add_url_rule("/staff",
                     view_func=views.staff_add_page, methods=['GET', 'POST'])
    app.add_url_rule("/facility",
                     view_func=views.facility_page, methods=['GET', 'POST'])
    app.add_url_rule("/facility/<id>",
                     view_func=views.facility_page, methods=['GET', 'POST'])
    app.add_url_rule("/staff/<staff_id>",
                     view_func=views.staff_add_page, methods=['GET', 'POST'])
    app.add_url_rule("/test",
                     view_func=views.test_page, methods=['GET', 'POST'])
    app.add_url_rule(
        "/instructors", view_func=views.instructors_page, methods=['GET'])
    app.add_url_rule("/instructors/add",
                     view_func=views.add_instructor_page, methods=['GET', 'POST'])
    app.add_url_rule("/instructor/<id>/edit",
                     view_func=views.edit_instructor_page, methods=['GET', 'POST'])
    app.add_url_rule("/courses", view_func=views.courses_page,
                     methods=["POST", "GET"])
    app.add_url_rule(
        "/courses/add", view_func=views.add_course_page, methods=['POST', 'GET'])
    app.add_url_rule("/course/<crn>/edit",
                     view_func=views.edit_course_page, methods=['POST', 'GET'])
    app.add_url_rule(
        "/course/<crn>", view_func=views.course_info_page, methods=['POST', 'GET'])
    app.add_url_rule("/my_courses", view_func=views.my_courses_page, methods=['GET'])
    db = init_db(db_url)
    app.config["db"] = db
    lm.init_app(app)
    lm.login_view = "login_page"

    @lm.user_loader
    def load_user(id):
        return db.get_user(id)

    return app


db_url = os.getenv("DATABASE_URL")
app = create_app(db_url)

if __name__ == "__main__":
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    debug = app.config.get("DEBUG")
    app.run(host=host, port=port, debug=debug)
