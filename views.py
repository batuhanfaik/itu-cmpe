from flask import current_app, render_template, request, redirect, url_for, abort, flash
from flask_login import login_required, logout_user, login_user

from campus import Campus
from faculty import Faculty
from forms import login_form
from person import Person
from passlib.hash import pbkdf2_sha256 as hash_machine


def landing_page():
    return render_template("index.html")


def login_page():
    # Here we use a class of some kind to represent and validate our
    # client-side form data. For example, WTForms is a library that will
    # handle this for us, and we use a custom LoginForm to validate.
    form = login_form()
    db = current_app.config["db"]
    if request.method == 'POST':
        if form.validate_on_submit():
            username = request.form['email']
            password = request.form['password']
            user = db.get_user(username)
            if user is None:
                flash('There is no such a user')
                form.errors['email'] = 'There is no such a user!'
            else:
                if hash_machine.verify(password, user.password):
                    login_user(user)
                    flash('Logged in successfully')
                else:
                    flash('Wrong password')
                    form.errors['password'] = 'Wrong password!'

            # if person is None:
            #     flash('Invalid username', 'error')
            #     return redirect(url_for('login'))
            # else:
            #     if(person.password == password):
            #         flash('Login succesfull')
            #     else:
            #         flash('Password is wrong!', 'error')
            #     login_user(person)
    return render_template('login.html', form=form)


@login_required
def logout_page():
    logout_user()
    return redirect(url_for("landing_page"))


def people_page():
    db = current_app.config["db"]
    people = db.get_people()
    if request.method == "GET":
        return render_template("people.html", people=sorted(people))
    else:
        form_tr_id = request.form["tr_id"]
        form_name = request.form["name"]
        form_surname = request.form["surname"]
        form_phone = request.form["phone"]
        form_email = request.form["email"]
        form_pwd = request.form["pwd"]
        form_pwd = hash_machine.hash(form_pwd)
        form_category = int(request.form["category"])
        form_mfname = request.form["mfname"]
        form_ffname = request.form["ffname"]
        form_gender = request.form["gender"]
        form_bcity = request.form["bcity"]
        form_bdate = request.form["bdate"]
        form_id_regcity = request.form["id_regcity"]
        form_id_regdist = request.form["id_regdist"]
        person = Person(form_tr_id, form_name, form_surname, form_phone, form_email, form_pwd,
                        form_category, form_mfname, form_ffname, form_gender, form_bcity,
                        form_bdate, form_id_regcity, form_id_regdist)
        db = current_app.config["db"]
        person_tr_id = db.add_person(person)
        people = db.get_people()
        return render_template("people.html", people=sorted(people))


def person_page(tr_id):
    db = current_app.config["db"]
    person = db.get_person(tr_id)
    if person is None:
        abort(404)
    if request.method == "GET":
        return render_template("person.html", person=person)
    else:
        if request.form["update_button"] == "update":
            form_tr_id = request.form["tr_id"]
            form_name = request.form["name"]
            form_surname = request.form["surname"]
            form_phone = request.form["phone"]
            form_email = request.form["email"]
            form_pwd = request.form["pwd"]
            form_category = int(request.form["category"])
            form_mfname = request.form["mfname"]
            form_ffname = request.form["ffname"]
            form_gender = request.form["gender"]
            form_bcity = request.form["bcity"]
            form_bdate = request.form["bdate"]
            form_id_regcity = request.form["id_regcity"]
            form_id_regdist = request.form["id_regdist"]
            person = Person(form_tr_id, form_name, form_surname, form_phone, form_email, form_pwd,
                            form_category, form_mfname, form_ffname, form_gender, form_bcity,
                            form_bdate, form_id_regcity, form_id_regdist)
            db = current_app.config["db"]
            db.update_person(person, tr_id)
            return redirect(url_for("person_page", tr_id=person.tr_id))
        elif request.form["update_button"] == "delete":
            db.delete_person(tr_id)
            people = db.get_people()
            return redirect(url_for("people_page", people=sorted(people)))


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
            edited_campus = Campus(
                campus_name, campus_location, campuses.get(campus_id).faculties)
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
            db.add_faculty(int(campus_id), Faculty(
                faculty_name, faculty_shortened_name))
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
