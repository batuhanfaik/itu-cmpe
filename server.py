from flask import Flask

import views
from database import Database
from campus import Campus

def create_app():
    app = Flask(__name__)
    app.add_url_rule("/", view_func=views.manage_campusses,methods=['GET', 'POST',"DELETE"])
    db = Database()
    app.config["db"] = db
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8082, debug=True)

