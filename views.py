from os import getenv

from flask import current_app, render_template, request, redirect, url_for, abort, flash
from flask_login import login_required, logout_user, login_user, current_user
from passlib.hash import pbkdf2_sha256 as hash_machine
from psycopg2 import errors, Error
from werkzeug.utils import secure_filename

import dbinit
from assistant import Assistant
from campus import Campus
from classroom import Classroom
from course import Course
from facility import Facility
from faculty import Faculty
from forms import login_form, InstructorForm, ClassroomForm, CourseForm, SelectCourseForm
from instructor import Instructor
from person import Person
from staff import Staff
from staff_facil import Staff_facil
from student import Student


def landing_page():
    return render_template("index.html")


def login_page():
    # Here we use a class of some kind to represent and validate our
    # client-side form data. For example, WTForms is a library that will
    # handle this for us, and we use a custom LoginForm to validate.
    if current_user.is_authenticated:
        return redirect(url_for('landing_page'))

    form = login_form()
    db = current_app.config["db"]

    if request.method == 'POST':
        if form.validate_on_submit():
            username = request.form['username']
            password = request.form['password']
            user = db.get_user(username)
            if user is None:
                flash('There is no such a user')
                form.errors['username'] = 'There is no such a user!'
            else:
                if hash_machine.verify(password, user.password):
                    login_user(user)
                    flash('Logged in successfully')
                    return redirect(url_for('landing_page'))
                else:
                    flash('Wrong password')
                    form.errors['password'] = 'Wrong password!'
        return redirect(url_for('login_page'))
    return render_template('login.html', form=form)


@login_required
def logout_page():
    logout_user()
    return redirect(url_for("landing_page"))


def validate_people_form(form):
    form.data = {}
    form.errors = {}
    db = current_app.config["db"]

    form_tr_id = form.get("tr_id")
    if db.get_person(form_tr_id):
        form.errors["tr_id"] = "There exists a person with the given TR ID."
    else:
        form.data["tr_id"] = form_tr_id

    form_email = form.get("email")
    if db.get_person_email(form_email):
        form.errors["email"] = "There exists a person with the given email address."
    else:
        form.data["email"] = form_email

    form.data["name"] = form.get("name")
    form.data["surname"] = form.get("surname")
    form.data["phone"] = form.get("phone")
    form.data["mfname"] = form.get("mfname")
    form.data["ffname"] = form.get("ffname")
    form.data["bcity"] = form.get("bcity")
    form.data["bdate"] = form.get("bdate")
    form.data["id_regcity"] = form.get("id_regcity")
    form.data["id_regdist"] = form.get("id_regdist")

    return len(form.errors) == 0


def people_page():
    db = current_app.config["db"]
    people = db.get_people()
    if request.method == "GET":
        return render_template("people.html", people=sorted(people), values=request.form)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
        valid = validate_people_form(request.form)
        if not valid:
            return render_template("people.html", people=sorted(people),
                                   values=request.form)
        form_tr_id = request.form.data["tr_id"]
        form_name = request.form.data["name"]
        form_surname = request.form.data["surname"]
        form_phone = request.form.data["phone"]
        form_email = request.form.data["email"]
        form_pwd = request.form["pwd"]
        form_pwd = hash_machine.hash(form_pwd)
        form_category = int(request.form["category"])
        form_mfname = request.form.data["mfname"]
        form_ffname = request.form.data["ffname"]
        form_gender = request.form["gender"]
        form_bcity = request.form.data["bcity"]
        form_bdate = request.form.data["bdate"]
        form_id_regcity = request.form.data["id_regcity"]
        form_id_regdist = request.form.data["id_regdist"]

        if "photo" not in request.files:
            flash('No file part')
            filename, file_extension, photo_data = "", "", ""
        else:
            photo = request.files["photo"]
            filename = secure_filename(photo.filename)
            file_extension = filename.split(".")[-1]
            filename = filename.split(".")[0]
            photo_data = request.files['photo'].read()

        person = Person(form_tr_id, form_name, form_surname, form_phone, form_email, form_pwd,
                        form_category, form_mfname, form_ffname, form_gender, form_bcity,
                        form_bdate, form_id_regcity, form_id_regdist, filename, file_extension,
                        photo_data)
        db = current_app.config["db"]
        person_tr_id = db.add_person(person)
        #if(form_category == 1):
            #new_staff = Staff(id=form_tr_id)
            #try:
                #db.add_staff(new_staff)
            #except Error as e:
                #flash('Staff could not created!')
                #print(type(e))
                #flash(type(e))
        people = db.get_people()
        return render_template("people.html", people=sorted(people), values={})


def person_page(tr_id):
    db = current_app.config["db"]
    person = db.get_person(tr_id)
    if person is None:
        abort(404)
    if request.method == "GET":
        return render_template("person.html", person=person)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
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

            if "photo" not in request.files:
                flash('No file part')
                filename, file_extension, photo_data = "", "", ""
            else:
                photo = request.files["photo"]
                filename = secure_filename(photo.filename)
                file_extension = filename.split(".")[-1]
                filename = filename.split(".")[0]
                photo_data = request.files['photo'].read()

            person = Person(form_tr_id, form_name, form_surname, form_phone, form_email, form_pwd,
                            form_category, form_mfname, form_ffname, form_gender, form_bcity,
                            form_bdate, form_id_regcity, form_id_regdist, filename, file_extension,
                            photo_data)
            db = current_app.config["db"]
            db.update_person(person, tr_id)
            return redirect(url_for("person_page", tr_id=person.tr_id))
        elif request.form["update_button"] == "delete":
            db.delete_person(tr_id)
            people = db.get_people()
            return redirect(url_for("people_page", people=sorted(people), values={}))


def validate_students_form(form):
    form.data = {}
    form.errors = {}
    db = current_app.config["db"]

    form_tr_id = form.get("tr_id")
    if db.get_student(form_tr_id):
        form.errors["tr_id"] = "There exists a student with the given TR ID."
    else:
        form.data["tr_id"] = form_tr_id
    form_student_id = form.get("student_id")
    if db.get_student_via_student_id(form_student_id):
        form.errors["student_id"] = "There exists a student with the given student ID."
    else:
        form.data["student_id"] = form_student_id

    form.data["faculty_id"] = form.get("faculty_id")
    form.data["department_id"] = form.get("department_id")
    form.data["semester"] = form.get("semester")
    form.data["grade"] = form.get("grade")
    form.data["gpa"] = form.get("gpa")
    form.data["credits_taken"] = form.get("credits_taken")
    form.data["minor"] = form.get("minor")

    return len(form.errors) == 0


def students_page():
    db = current_app.config["db"]
    students = db.get_students()
    if request.method == "GET":
        return render_template("students.html", students=sorted(students), values=request.form)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
        valid = validate_students_form(request.form)
        if not valid:
            return render_template("students.html", students=sorted(students), values=request.form)
        form_tr_id = request.form.data["tr_id"]
        form_faculty_id = request.form.data["faculty_id"]
        form_department_id = request.form.data["department_id"]
        form_student_id = request.form.data["student_id"]
        form_semester = request.form.data["semester"]
        form_grade = request.form.data["grade"]
        form_gpa = request.form.data["gpa"]
        form_credits_taken = request.form.data["credits_taken"]
        form_minor = request.form.data["minor"]

        student = Student(form_tr_id, form_faculty_id, form_department_id, form_student_id,
                          form_semester, form_grade, form_gpa, form_credits_taken, form_minor)
        db = current_app.config["db"]
        student_tr_id = db.add_student(student)
        students = db.get_students()
        return render_template("students.html", students=sorted(students), values={})


def student_page(tr_id):
    db = current_app.config["db"]
    student = db.get_student(tr_id)
    if student is None:
        abort(404)
    if request.method == "GET":
        return render_template("student.html", student=student)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
        if request.form["update_button"] == "update":
            form_tr_id = request.form["tr_id"]
            form_faculty_id = request.form["faculty_id"]
            form_department_id = request.form["department_id"]
            form_student_id = request.form["student_id"]
            form_semester = request.form["semester"]
            form_grade = request.form["grade"]
            form_gpa = request.form["gpa"]
            form_credits_taken = request.form["credits_taken"]
            form_minor = request.form["minor"]

            student = Student(form_tr_id, form_faculty_id, form_department_id, form_student_id,
                              form_semester, form_grade, form_gpa, form_credits_taken, form_minor)
            db = current_app.config["db"]
            db.update_student(student, tr_id)
            return redirect(url_for("student_page", tr_id=student.tr_id))
        elif request.form["update_button"] == "delete":
            db.delete_student(tr_id)
            students = db.get_students()
            return redirect(url_for("students_page", students=sorted(students), values={}))


def validate_assistants_form(form):
    form.data = {}
    form.errors = {}
    db = current_app.config["db"]

    form_tr_id = form.get("tr_id")
    if db.get_student(form_tr_id):
        form.errors["tr_id"] = "There exists an assistant with the given TR ID."
    else:
        form.data["tr_id"] = form_tr_id
    form_assistant_id = form.get("assistant_id")
    if db.get_assistant_via_assistant_id(form_assistant_id):
        form.errors["assistant_id"] = "There exists an assistant with the given assistant ID."
    else:
        form.data["assistant_id"] = form_assistant_id

    form.data["faculty_id"] = form.get("faculty_id")
    form.data["supervisor"] = form.get("supervisor")
    form.data["bachelors"] = form.get("bachelors")
    form.data["grad_gpa"] = form.get("grad_gpa")
    form.data["research_area"] = form.get("research_area")

    return len(form.errors) == 0


def assistants_page():
    db = current_app.config["db"]
    assistants = db.get_assistants()
    if request.method == "GET":
        return render_template("assistants.html", assistants=sorted(assistants),
                               values=request.form)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
        valid = validate_assistants_form(request.form)
        if not valid:
            return render_template("assistants.html", assistants=sorted(assistants),
                                   values=request.form)
        form_tr_id = request.form.data["tr_id"]
        form_faculty_id = request.form.data["faculty_id"]
        form_supervisor = request.form.data["supervisor"]
        form_assistant_id = request.form.data["assistant_id"]
        form_bachelors = request.form.data["bachelors"]
        form_degree = request.form["degree"]
        form_grad_gpa = request.form.data["grad_gpa"]
        form_research_area = request.form.data["research_area"]
        form_office_day = request.form["office_day"]
        form_office_hour_start = request.form["office_hour_start"]
        form_office_hour_end = request.form["office_hour_end"]

        assistant = Assistant(form_tr_id, form_faculty_id, form_supervisor, form_assistant_id,
                              form_bachelors, form_degree, form_grad_gpa, form_research_area,
                              form_office_day, form_office_hour_start, form_office_hour_end)
        db = current_app.config["db"]
        assistant_tr_id = db.add_assistant(assistant)
        assistants = db.get_assistants()
        return render_template("assistants.html", assistants=sorted(assistants), values={})


def assistant_page(tr_id):
    db = current_app.config["db"]
    assistant = db.get_assistant(tr_id)
    if assistant is None:
        abort(404)
    if request.method == "GET":
        return render_template("assistant.html", assistant=assistant)
    else:
        if current_user.role != 'admin':
            return redirect(url_for("landing_page"))
        if request.form["update_button"] == "update":
            form_tr_id = request.form["tr_id"]
            form_faculty_id = request.form["faculty_id"]
            form_department_id = request.form["department_id"]
            form_student_id = request.form["student_id"]
            form_semester = request.form["semester"]
            form_grade = request.form["grade"]
            form_gpa = request.form["gpa"]
            form_credits_taken = request.form["credits_taken"]
            form_minor = request.form["minor"]

            assistant = Assistant(form_tr_id, form_faculty_id, form_department_id, form_student_id,
                                  form_semester, form_grade, form_gpa, form_credits_taken,
                                  form_minor)
            db = current_app.config["db"]
            db.update_assistant(assistant, tr_id)
            return redirect(url_for("assistant_page", tr_id=assistant.tr_id))
        elif request.form["update_button"] == "delete":
            db.delete_assistant(tr_id)
            assistants = db.get_assistants()
            return redirect(url_for("assistants_page", assistants=sorted(assistants), values={}))


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


@login_required
def reset_db():
    db_url = getenv("DATABASE_URL")
    if current_user.is_admin:
        dbinit.reset_db(db_url)
    return redirect(url_for("landing_page"))

# faati's pages #
# classroom pages #
# classrooms_page is inside faculty_detailed page #


def add_classroom_page(faculty_id):
    form = ClassroomForm()
    if form.validate_on_submit():
        db = current_app.config['db']
        capacity = form.data['capacity']
        has_projection = form.data['has_projection']
        door_number = form.data['door_number']
        floor = form.data['floor']
        renewed = form.data['renewed']
        board_count = form.data['board_count']
        air_conditioner = form.data['air_conditioner']
        classroom = db.get_classroom_by_door_and_faculty(faculty_id, door_number)
        if classroom is not None:
            return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Add Classroom",
                                   error="There exists a classroom with this door number in this faculty!")
        try:
            db.add_classroom(Classroom(None, capacity, has_projection, door_number, floor, renewed,
                                   board_count, air_conditioner, faculty_id))
            return redirect(url_for("faculty_detailed", faculty_id=faculty_id))
        except Error as e:
            str_e = str(e)
            error = type(e).__name__ + '----' + str_e
            if isinstance(e, errors.UniqueViolation):
                error = "This classroom already exists in the given building"
            return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Add Classroom",
                                   error=error)
    return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Add Classroom", error=None)


def edit_classroom_page(faculty_id, id):
    error = ""
    form = ClassroomForm()
    db = current_app.config['db']
    if form.validate_on_submit():
        if request.form['btn'] == 'update':
            capacity = form.data['capacity']
            has_projection = form.data['has_projection']
            door_number = form.data['door_number']
            floor = form.data['floor']
            renewed = form.data['renewed']
            board_count = form.data['board_count']
            air_conditioner = form.data['air_conditioner']
            classroom = db.get_classroom_by_door_and_faculty(faculty_id, door_number)
            if (classroom is not None) and (classroom.id != int(id)):
                return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Update Classroom",
                                       error="There exists a classroom with this door number in this faculty!")
            try:
                db.update_classroom(id, Classroom(None, capacity, has_projection, door_number, floor, renewed,
                                       board_count, air_conditioner, faculty_id))
                return redirect(url_for("faculty_detailed", faculty_id=faculty_id))
            except Error as e:
                error = type(e).__name__ + '----' + str(e)
                str_e = str(e)
                if isinstance(e, errors.UniqueViolation):
                    error = "This classroom already exists in the given building"
    if request.method == 'POST' and request.form['btn'] == 'delete':
        try:
            db.delete_classroom(id)
            return redirect(url_for("faculty_detailed", faculty_id=faculty_id))
        except Error as e:
            error = type(e).__name__ + '----' + str(e)
            if isinstance(e, errors.ForeignKeyViolation):
                str_e = str(e)
                if 'course' in str_e:
                    error = "There are courses given in this classroom!"
            pass
    classroom = db.get_classroom(id)
    form.capacity.data = classroom.capacity
    form.has_projection.data = classroom.has_projection
    form.door_number.data = classroom.door_number
    form.floor.data = classroom.floor
    form.renewed.data = classroom.renewed
    form.board_count.data = classroom.board_count
    form.air_conditioner.data = classroom.air_conditioner
    return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Update Classroom",
                           error=error)


# course pages#
def courses_page():
    db = current_app.config["db"]
    courses = db.get_all_courses()
    return render_template("courses.html", courses=courses)


@login_required
def my_courses_page():
    if current_user.role != 'student' and current_user.role != 'instructor':
        return redirect(url_for("landing_page"))
    db = current_app.config['db']
    courses = []
    if current_user.student_id is not None:
        courses = db.get_courses_taken_by_student(current_user.student_id)
    elif current_user.instructor_id is not None:
        courses = db.get_courses_by_instructor_id(current_user.instructor_id)
    return render_template("courses.html", courses=courses)

@login_required
def add_course_page():
    if current_user.role != 'admin':
        return redirect(url_for("landing_page"))
    form = CourseForm()
    if form.validate_on_submit():
        db = current_app.config['db']
        args = []
        for key, value in form.data.items():
            if key != 'csrf_token':
                args.append(value)
        course = Course(*args)
        if not db.is_classroom_available(course.start_time, course.end_time, course.day,
                                                        course.classroom_id):
            error = "There is already a course given in that classroom at that time!"
            return render_template("edit_course.html", form=form, error=error, title="Add Course")
        if not db.is_instructor_available(course.start_time, course.end_time, course.day, course.instructor_id):
            error = "The instructor already has a course at that time!"
            return render_template("edit_course.html", form=form, error=error, title="Add Course")
        try:
            db.add_course(course)
            return redirect(url_for('courses_page'))
        except Error as e:
            error = type(e).__name__ + '----' + str(e)
            str_e = str(e)
            if isinstance(e, errors.UniqueViolation):
                error = "This course already exists"
            if isinstance(e, errors.ForeignKeyViolation):
                if 'classroom' in str_e:
                    error = "There is no classroom with given id"
                if 'department' in str_e:
                    error = "There is no department with given id"
                if 'instructor' in str_e:
                    error = "There is no instructor with given id"
            return render_template("edit_course.html", form=form, error=error, title="Add Course")
    return render_template("edit_course.html", form=form, error=None, title="Add Course")


@login_required
def select_courses_page():
    if current_user.role != 'student':
        return redirect(url_for("landing_page"))
    db = current_app.config['db']
    form = SelectCourseForm()
    results = []
    if form.validate_on_submit():
        if request.form['btn'] == 'add':
            crn_list = []
            for key, value in form.data.items():
                if key != 'csrf_token' and value != 0:
                    crn_list.append(str(value))
            for crn in crn_list:
                result = {'crn': crn}
                try:
                    course = db.get_course(crn)
                    if course is not None:
                        if db.student_can_take_course(current_user.student_id, course):
                            db.add_taken_course(current_user.student_id, crn)
                            result['result'] = "You have been added to this course!"
                            db.update_course_enrollment(crn)
                        else:
                            result['result'] = "This course conflicts with another course you have"
                    else:
                        result['result'] = "This course does not exists"
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
                    str_e = str(e)
                    if isinstance(e, errors.UniqueViolation):
                        error = "You already have this course"
                    if isinstance(e, errors.ForeignKeyViolation):
                        if 'course' in str_e:
                            error = "This CRN does not belongs to any course"
                    result['result'] = error
                results.append(result)
        else:
            crn_list = []
            for key, value in form.data.items():
                if key != 'csrf_token' and value != 0:
                    crn_list.append(str(value))
            for crn in crn_list:
                result = {'crn': crn}
                try:
                    if db.get_course(crn) is not None:
                        db.delete_taken_course(current_user.student_id, crn)
                        result['result'] = "Successfully dropped course"
                        db.update_course_enrollment(crn)
                    else:
                        result['result'] = "This CRN does not belongs to any course"
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
                    str_e = str(e)
                    if isinstance(e, errors.UniqueViolation):
                        error = "This CRN does not belongs to any course"
                    if isinstance(e, errors.ForeignKeyViolation):
                        if 'course' in str_e:
                            error = "This CRN does not belongs to any course"
                    result['result'] = error
                results.append(result)
    return render_template("select_courses.html", form=form, results=results, error=None, title="Add/Drop Courses")



@login_required
def edit_course_page(crn):
    if current_user.role != 'admin':
        return redirect(url_for("landing_page"))
    error = ""
    db = current_app.config["db"]
    course = db.get_course(crn)
    form = CourseForm(data=course.__dict__)
    form.crn(readonly=True)
    if form.validate_on_submit():
        if request.form['btn'] == 'update':
            args = []
            for key, value in form.data.items():
                if key != 'csrf_token':
                    args.append(value)
            course = Course(*args)
            course.crn = crn
            try:
                db.update_course(crn, course)
                return redirect(url_for("courses_page"))
            except Error as e:
                error = type(e).__name__ + '----' + str(e)
                str_e = str(e)
                if isinstance(e, errors.ForeignKeyViolation):
                    if 'classroom' in str_e:
                        error = "There is no classroom with given id"
                    if 'department' in str_e:
                        error = "There is no department with given id"
                    if 'instructor' in str_e:
                        error = "There is no instructor with given id"
                pass
    if request.method == 'POST' and request.form['btn'] == 'delete':
        db.delete_course(crn)
        return redirect(url_for("courses_page"))
    return render_template("edit_course.html", form=form, error=error, title="Edit Course")


# instructor pages#
@login_required
def instructors_page():
    if current_user.role != 'admin':
        return redirect(url_for("landing_page"))
    db = current_app.config["db"]
    instructors = db.get_all_instructors()
    return render_template("instructors.html", instructors=instructors)


@login_required
def add_instructor_page():
    if current_user.role != 'admin':
        return redirect(url_for("landing_page"))
    form = InstructorForm()
    if form.validate_on_submit():
        db = current_app.config["db"]
        id = None
        tr_id = form.data['tr_id']
        department_id=form.data['department_id']
        faculty_id = form.data['faculty_id']
        specialization = form.data['specialization']
        bachelors = form.data['bachelors']
        masters = form.data['masters']
        doctorates = form.data['doctorates']
        room_id = form.data['room_id']
        instructor = Instructor(id, tr_id, department_id, faculty_id, specialization,
                                bachelors, masters, doctorates, room_id)
        try:
            db.add_instructor(instructor)
        except Error as e:
            error = type(e).__name__ + "-----" + str(e)
            if isinstance(e, errors.UniqueViolation):
                error = "An instructor with this TR ID already exists"
            if isinstance(e, errors.ForeignKeyViolation):
                str_e = str(e)
                if 'tr_id' in str_e:
                    error = "No people exists with this TR ID"
                elif 'faculty_id' in str_e:
                    error = "No faculty exists with this Faculty ID"
                elif 'department_id' in str_e:
                    error = "No department exists with this Department ID"
            return render_template("edit_instructor.html", form=form, title="Add Instructor",
                                   error=error)
        return redirect(url_for("instructors_page"))
    return render_template("edit_instructor.html", form=form, title="Add Instructor", error=None)


@login_required
def edit_instructor_page(id):
    if current_user.role != 'admin':
        return redirect(url_for("landing_page"))
    error = ""
    db = current_app.config["db"]
    form = InstructorForm()
    if form.validate_on_submit():
        if request.form['btn'] == 'update':
            tr_id = form.data['tr_id']
            department_id = form.data['department_id']
            faculty_id = form.data['faculty_id']
            specialization = form.data['specialization']
            bachelors = form.data['bachelors']
            masters = form.data['masters']
            doctorates = form.data['doctorates']
            room_id = form.data['room_id']
            instructor = Instructor(id, tr_id, department_id, faculty_id, specialization,
                                    bachelors, masters, doctorates, room_id)
            try:
                db.update_instructor(id, instructor)
                return redirect(url_for("instructors_page"))
            except Error as e:
                error = type(e).__name__ + "-----" + str(e)
                if isinstance(e, errors.UniqueViolation):
                    error = "An instructor with this TR ID already exists"
                if isinstance(e, errors.ForeignKeyViolation):
                    str_e = str(e)
                    if 'tr_id' in str_e:
                        error = "No people exists with this TR ID"
                    elif 'faculty_id' in str_e:
                        error = "No faculty exists with this Faculty ID"
                    elif 'department_id' in str_e:
                        error = "No department exists with this Department ID"
                pass
    if request.method == 'POST' and request.form['btn'] == 'delete':
        try:
            db.delete_instructor(id)
            return redirect(url_for("instructors_page"))
        except Error as e:
            error = type(e).__name__ + '----' + str(e)
            if isinstance(e, errors.ForeignKeyViolation):
                str_e = str(e)
                if 'course' in str_e:
                    error = "There are courses given by this instructor! It can not be deleted!"
                elif 'assistant' in str_e:
                    error = "There are assistants supervised by this instructor! It can not be deleted!"
            pass
    instructor = db.get_instructor(id)
    form.tr_id.data = instructor.tr_id
    form.room_id.data = instructor.room_id
    form.doctorates.data = instructor.doctorates
    form.masters.data = instructor.masters
    form.bachelors.data = instructor.bachelors
    form.specialization.data = instructor.specialization
    form.department_id.data = instructor.department_id
    form.faculty_id.data = instructor.faculty_id
    return render_template("edit_instructor.html", form=form, title="Update Instructor", error=error)


def test_page():
    return render_template("test.html")


def course_info_page(crn):
    db = current_app.config["db"]
    taken_course_students = db.get_taken_course_by_crn(crn)
    db.update_course_enrollment(crn)
    course = db.get_course(crn)
    department= db.get_department(course.department_id)
    faculty= db.get_faculty(department.faculty_id)
    students = []
    give_permission_to_see = False
    if(current_user.is_authenticated):
        instructor = db.get_instructor_via_tr_id(current_user.tr_id)
        give_permission_to_see = False
        if (not instructor is None):
            is_this_course = db.get_course_via_instructor_id(instructor.id)
            if (not is_this_course is None):
                give_permission_to_see = True
    for student in taken_course_students:
        std = db.get_student_via_student_id(student.student_id)
        std.grade = student.grade
        pers = db.get_person(std.tr_id)
        student_name = pers.name
        student_last_name = pers.surname
        std.tr_id = pers.tr_id
        std.name = student_name + " "+student_last_name
        students.append(std)
    context={
        'students' : students,
        'course' : course,
        'department':department,
        'faculty':faculty,
        'give_permission_to_see':give_permission_to_see
    }
    if(request.method == "POST" and "redirect_course_edit_page" in request.form):
        return redirect(url_for('edit_course_page',crn=crn))
    if(request.method == "POST" and "post_grade_form" in request.form):
        for taken_course in taken_course_students:
            strm = 'std'+str(taken_course.student_id)
            print(request.form)
            taken_course.grade = request.form[strm]
            if(taken_course.grade!="None"):
                db.update_taken_course(taken_course.id,taken_course)
        return redirect(url_for('course_info_page', crn = crn))
    return render_template("course_inf.html",context= context)

def validation_staff(form):
    form.data = {}
    form.errors = {}
    db = current_app.config["db"]

    form_id = form.get("id")
    if db.get_staff(form_id):
        form.errors["id"] = "This staff is already registered with the given id."
        flash('This staff is already registered with the given id')
    elif form.get("id") == 0 or form.get("id") ==None:
        form.errors["id"] = "ID cannot be empty."
        flash('ID cannot be empty.')
    elif form.get("hire_date") == 0:
        form.errors["hire_date"] = "Hire Date cannot be empty."
        flash('Hire Date cannot be empty.')
    elif form.get("social_sec_no") == 0:
        form.errors["social_sec_no"] = "Social Security Number cannot be empty."
        flash('Social Security Number cannot be empty')
    elif not db.get_person(form_id):
        form.errors["id"] = "There is no Person with the given ID."
        flash('No people exists with this TR ID')


    else:
        form.data["id"] = form.get("id")
        form.data["manager_name"] = form.get("manager_name")
        form.data["absences"] = form.get("absences")
        form.data["hire_date"] = form.get("hire_date")
        form.data["authority_lvl"] = form.get("authority_lvl")
        form.data["department"] = form.get("department")
        form.data["social_sec_no"] = form.get("social_sec_no")
    return len(form.errors) == 0



def staff_add_page():
    db = current_app.config["db"]
    all_staff = db.get_all_staff()
    if request.method == "GET":
        return render_template("staff.html",staffs = all_staff, values=request.form)

    elif 'search_staff' in request.form:
        print("Searching staff.. id:",request.form.get("staff-id"))
        found_staff = db.get_staff(request.form.get("staff-id"))
        person_info = db.get_person(request.form.get("staff-id"))
        if found_staff is None:
            flash('No staff has been found.')
            return render_template("staff.html", staffs=all_staff,
                                   values=request.form)
        else:
            flash('Staff found!')
            return render_template("staff_search.html", staff=found_staff,staff_id = found_staff.id,
                                   values=request.form,person_info= person_info)
    elif 'delete_staff' in request.form:


        staff_id = request.form["staff_id"]
        print("Delete staff!", staff_id)
        db.delete_staff(int(staff_id))
        flash('Staff Deleted!')
        all_staff = db.get_all_staff()
        return render_template("staff.html",staffs=all_staff,
                               values=request.form)


    elif 'update_staff' in request.form:

        print("UPDATEEEE")
        old_staff_id = request.form["staff_id"]
        manager_name = request.form.get("manager_name")
        absences = request.form.get("absences")
        hire_date = request.form.get("hire_date")
        authority = request.form.get("authority_lvl")
        department = request.form.get("department")
        social_sec = request.form.get("social_sec_no")

        new_staff = Staff(id=old_staff_id, manager_name=manager_name, absences=absences, hire_date=hire_date,
                          social_sec_no=social_sec, department=department, authority_lvl=authority)
        print(new_staff.id, new_staff.manager_name, new_staff.absences,new_staff.hire_date, new_staff.social_sec_no,new_staff.department)
        db.update_staff(new_staff)

        flash('Staff Updated!')
        all_staff = db.get_all_staff()
        return render_template("staff.html",staffs=all_staff,
                               values=request.form)

    elif 'more_info' in request.form:
        s_id = request.form["staff_id"]
        staff_facilities = db.get_facility_from_staff(s_id)
        print("\n staff id: ",s_id," facility length", len(staff_facilities))
        the_staff = db.get_staff(s_id)
        facils = []
        for SF in staff_facilities:
            facility_ = db.get_facility(SF.facility_id)
            facils.append(facility_)
        if(len(facils) is None ):
            flash('No facility information is given for this Staff')
        return render_template("staff_facility.html", staff = the_staff, facilities = facils,
                                 staff_facils = staff_facilities,values=request.form)
    elif 'add_staff_facil' in request.form:
        #Check validation
        #add Delete/update
        s_id = request.form["staff_id"]
        staff_facilities = db.get_facility_from_staff(s_id)
        print("\n staff id: ", s_id, " facility length", len(staff_facilities))
        the_staff = db.get_staff(s_id)
        facils = []
        for SF in staff_facilities:
            facility_ = db.get_facility(SF.facility_id)
            print("\n id of current facility", facility_.id)
            facils.append(facility_)
        print("\nLength of facilities array: ", len(facils))
        title = request.form.get("title")
        from_date = request.form.get("from_date")
        to_date = request.form.get("to_date")
        salary = request.form.get("salary")
        facility_id = request.form.get("facility_id")
        staff_id = request.form.get("staff_id")
        duty = request.form.get("duty")
        new_SF = Staff_facil(title=title, from_date=from_date, to_date=to_date, salary=salary,
                             facility_id=facility_id, staff_id=staff_id, duty=duty)

        try:
            db.add_staff_facility(new_SF)
            flash('Staff-Facility Connection successfully added!')
            staff_facilities = db.get_facility_from_staff(s_id)
            print("\nLength of staff_facilities array:",len(staff_facilities))
            facils=[]
        except Error as e:
            flash('Staff-Facility Connection Not added!')
            print("\nERROR:",type(e))
            if isinstance(e, errors.UniqueViolation):
                flash('This connection already exists')
                return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                       staff_facils=staff_facilities, values=request.form,
                                           error="ID already exists")
            if isinstance(e, errors.ForeignKeyViolation):
                flash('Could not find the given staff Id or Facility Id' )
                return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                       staff_facils=staff_facilities, values=request.form,
                                           error="No ID")

            else:
                return render_template("staff_facility.html", staff = the_staff, facilities = facils,
                                 staff_facils = staff_facilities,values=request.form,
                                           error=type(e).__name__ + "-----" + str(e))
        for SF in staff_facilities:
            facility_ = db.get_facility(SF.facility_id)
            print("\n id of current facility", facility_.id)
            facils.append(facility_)
        return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                   staff_facils=staff_facilities, values=request.form)

    else:#Staff addition
        valid = validation_staff(request.form)
        if not valid:
            flash('Input NOT Valid!')
            return render_template("staff.html", staffs=all_staff,
                                   values=request.form)
        else:
            manager_name = request.form.data["manager_name"]
            staff_id = request.form.data["id"]
            absences = request.form.data["absences"]
            hire_date = request.form.data["hire_date"]
            authority = request.form.data["authority_lvl"]
            department = request.form.data["department"]
            social_sec = request.form.data["social_sec_no"]
            new_staff = Staff(id=staff_id,manager_name= manager_name,absences= absences,hire_date= hire_date,social_sec_no= social_sec,department= department,authority_lvl= authority)
            try:
                db.add_staff(new_staff)
                flash('Staff successfully added!')
            except Error as e:
                flash('Staff NOT added!')
                if isinstance(e, errors.UniqueViolation):
                    flash('A staff with this ID already exists')
                    return render_template("staff.html", form=request.form,staffs = all_staff,values=request.form,
                                           error="A staff with this ID already exists")
                if isinstance(e, errors.ForeignKeyViolation):
                    flash('No people exists with this TR ID')
                    return render_template("staff.html", form=request.form,staffs = all_staff,values=request.form,
                                           error="No people exists with this TR ID")

                else:
                    return render_template("staff.html", form=request.form,staffs = all_staff,values=request.form,
                                           error=type(e).__name__ + "-----" + str(e))
            return redirect(url_for("staff_add_page",staffs = all_staff,values=request.form))


def find_campus(campus_id):
     db = current_app.config["db"]
     campuses = db.get_campuses()
     for id,campus in campuses:
        if campus_id==id:
            return True
     return None

def validation_facility(form):
    form.data = {}
    form.errors = {}
    db = current_app.config["db"]

    form_id = form.get("id")
    form_campus_id = form.get("campus_id")

    if db.get_facility(form_id):
        form.errors["id"] = "This facility is already registered with the given id."
        flash('This facility is already registered with the given id')
    elif form.get("id") == 0 or form.get("id") ==None:
        form.errors["id"] = "ID cannot be empty."
        flash('ID cannot be empty.')
    elif form.get("campus_id") == 0:
        form.errors["campus_id"] = "Campus ID cannot be empty."
        flash('Campus ID cannot be empty.')
    elif form.get("name") == 0:
        form.errors["name"] = "Name cannot be empty."
        flash('Name cannot be empty')
    elif not find_campus(int(form_campus_id)):
        form.errors["id"] = "There is no Campus with the given Campus ID."
        flash('There is no Campus with the given Campus ID.')


    else:
        form.data["id"] = form.get("id")
        form.data["campus_id"] = form.get("campus_id")
        form.data["name"] = form.get("name")
        form.data["shortened_name"] = form.get("shortened_name")
        form.data["number_of_workers"] = form.get("number_of_workers")
        form.data["size"] = form.get("size")
        form.data["expenses"] = form.get("expenses")
    return len(form.errors) == 0



def facility_page():
    db = current_app.config["db"]
    all_facilities = db.get_all_facility()

    if request.method == "GET":
        return render_template("facility.html", values=request.form, facilities = all_facilities)


    elif 'facility_search' in request.form:
        facil = db.get_facility(request.form.get("facility_id"))
        if facil is None:
            flash('No facility has been found.')
            return render_template("facility.html",  facilities=all_facilities,
                                       values=request.form)
        else:
            flash('Facility found!')
            return render_template("facility_search.html", facility=facil, facility_id=facil.id,
                                    by_campus= 0, values=request.form)

    elif 'delete_facility' in request.form:
        f_id = request.form["facility_id"]
        db.delete_facility(int(f_id))
        flash('Facility Deleted!')
        all_f = db.get_all_facility()
        return render_template("facility.html", facilities=all_f,
                               values=request.form)

    elif 'search_facility_campus' in request.form:
        campus_id = request.form["find_campus_id"]
        campus= db.get_campus(campus_id)
        c_name=campus.name
        facilities = db.get_facility_from_campus(campus_id)
        if len(facilities) == 0:

            flash('There is no facility in this Campus.')
            return render_template("facility.html", facilities=all_facilities,
                                   values=request.form)
        return render_template("facility_search.html", facilities=facilities, campus_name = c_name,
                               by_campus = 1 ,values=request.form)

    elif 'update_facility' in request.form:

        id = request.form.get("id")
        campus_id = request.form.data["campus_id"]
        name = request.form.data["name"]
        short_name = request.form.data["shortened_name"]
        num_worker = request.form.data["number_of_workers"]
        size = request.form.data["size"]
        expense = request.form.data["expenses"]
        new_facil = Facility(id=id, campus_id=campus_id, name=name, shortened_name=short_name,
                             number_of_workers=num_worker, size=size, expenses=expense)

        db.update_facility(new_facil)

        flash('Facility Updated!')
        all_staff = db.get_all_staff()
        return render_template("staff.html",staffs=all_staff,
                               values=request.form)

    else:
        valid = validation_facility(request.form)
        if not valid:
            #flash('Input NOT Valid!')
            return render_template("facility.html", facilities=all_facilities,
                                   values=request.form)





        else:
            id = request.form.get("id")
            campus_id = request.form.data["campus_id"]
            name = request.form.data["name"]
            short_name = request.form.data["shortened_name"]
            num_worker = request.form.data["number_of_workers"]
            size = request.form.data["size"]
            expense = request.form.data["expenses"]
            new_facil = Facility(id=id, campus_id=campus_id, name=name, shortened_name=short_name,
                              number_of_workers=num_worker, size=size, expenses=expense)
            try:
                db.add_facility(new_facil)
                flash('Facility successfully added!')
                all_facilities = db.get_all_facility()
            except Error as e:
                flash('Facility NOT added!')
                if isinstance(e, errors.UniqueViolation):
                    flash('A Facility with this ID already exists')
                    return render_template("facility.html", form=request.form, facilities=all_facilities, values=request.form,
                                           error="A Facility with this ID already exists")
                if isinstance(e, errors.ForeignKeyViolation):
                    flash('No campus exists with this ID')
                    return render_template("facility.html", form=request.form,facilities=all_facilities, values=request.form,
                                           error="No campus exists with this ID")

                else:
                    return render_template("facility.html", form=request.form, facilities=all_facilities, values=request.form,
                                           error=type(e).__name__ + "-----" + str(e))
            return redirect(url_for("facility_page", facilities=all_facilities, values=request.form))