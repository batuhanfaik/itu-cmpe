from flask import current_app, render_template, request, redirect, url_for, abort

from campus import Campus
from faculty import Faculty


def landing_page():
    return render_template("index.html")


def login_page():
    error = None
    if request.method == 'GET':
        return render_template("login.html")
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = "Invalid credentials."
        else:
            return redirect(url_for('manage_campuses'))
    return render_template("login.html", error=error)


def people_page():
    db = current_app.config["db"]
    people = db.get_people()
    return render_template("people.html", people=sorted(people))


def person_page(tr_id):
    db = current_app.config["db"]
    person = db.get_person(tr_id)
    if person is None:
        abort(404)
    return render_template("person.html", person=person)


def manage_campuses():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    if request.method == "GET":
        return render_template("manage_campuses.html", campuses=campuses)
    else:
        if 'add_campus' in request.form:
            campus_name = request.form["campus_name"]
            campus_location = request.form["campus_location"]
            db.add_campus(Campus(campus_name, campus_location, []))
        elif 'edit_campus' in request.form:
            campus_id = request.form["campus_id"]
            campus_name = request.form["campus_name"]
            campus_location = request.form["campus_location"]
            edited_campus = Campus(campus_name, campus_location, campuses.get(campus_id).faculties)
            edited_campus.set_campus_id(int(campus_id))
            db.edit_campus(edited_campus)
        elif 'delete_campus' in request.form:
            db = current_app.config["db"]
            campuses = db.get_campuses()
            campus_id = request.form["campus_id"]
            db.delete_campus(int(campus_id))
        elif 'add_faculty' in request.form:
            db = current_app.config["db"]
            campuses = db.get_campuses()
            campus_id = request.form["campus_id"]
            faculty_name = request.form["faculty_name"]
            faculty_shortened_name = request.form["faculty_shortened_name"]
            db.add_faculty(int(campus_id), Faculty(faculty_name, faculty_shortened_name))
        elif 'delete_faculty' in request.form:
            db = current_app.config["db"]
            campuses = db.get_campuses()
            campus_id = request.form["campus_id"]
            faculty_id = request.form["faculty_id"]
            db.delete_faculty(int(campus_id), int(faculty_id))
        elif 'edit_faculty' in request.form:
            campus_id = request.form["campus_id"]
            faculty_id = request.form["faculty_id"]
            faculty_name = request.form["faculty_name"]
            faculty_shortened_name = request.form["faculty_shortened_name"]
            edited_faculty = Faculty(faculty_name, faculty_shortened_name)
            edited_faculty.set_faculty_id(int(faculty_id))
            db.edit_faculty(edited_faculty)

        return redirect(url_for("manage_campuses", campuses=campuses))
