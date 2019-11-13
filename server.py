from flask import Flask

import views
from database import Database
from person import Person


def create_app():
    app = Flask(__name__, static_url_path="/static/")

    app.config.from_object("settings")
    app.add_url_rule("/", view_func=views.landing_page, methods=['GET', 'POST'])
    app.add_url_rule("/login", view_func=views.login_page, methods=['GET', 'POST'])
    app.add_url_rule("/people", view_func=views.people_page, methods=['GET', 'POST'])
    app.add_url_rule("/people/<tr_id>", view_func=views.person_page, methods=['GET', 'POST'])
    app.add_url_rule("/campuses", view_func=views.manage_campuses,
                     methods=['GET', 'POST', "DELETE"])
    db = init_db()
    app.config["db"] = db
    return app


def init_db():
    db = Database()
    db.add_person(Person("11111111110", "Ahmet", "Mehmet"))
    return db


app = create_app()

if __name__ == "__main__":
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    debug = app.config.get("DEBUG")
    app.run(host=host, port=port, debug=debug)
