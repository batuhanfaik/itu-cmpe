from flask import current_app, render_template, request, redirect, url_for, abort, flash
from flask_login import login_required, logout_user, login_user, current_user
from passlib.hash import pbkdf2_sha256 as hash_machine
from werkzeug.utils import secure_filename
from os import getenv
from psycopg2 import errors, Error
import dbinit

from assistant import Assistant
from campus import Campus
from faculty import Faculty
from forms import login_form, InstructorForm, ClassroomForm, CourseForm
from person import Person
from student import Student
from instructor import Instructor
from staff import Staff
from classroom import Classroom
from course import Course
from facility import Facility

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
        db.add_classroom(Classroom(None, capacity, has_projection, door_number, floor, renewed,
                                   board_count, air_conditioner, faculty_id))
        return redirect(url_for("faculty_detailed", faculty_id=faculty_id))
    return render_template("edit_classroom.html", form=form, faculty_id=faculty_id, title="Add Classroom", error=None)


def update_classroom_page(faculty_id, id, error=None):
    form = ClassroomForm()
    db = current_app.config['db']
    if form.validate_on_submit():
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
        db.update_classroom(id, Classroom(None, capacity, has_projection, door_number, floor, renewed,
                                   board_count, air_conditioner, faculty_id))
        return redirect(url_for("faculty_detailed", faculty_id=faculty_id))
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


def delete_classroom(faculty_id, id):
    db = current_app.config['db']
    try:
        db.delete_classroom(id)
    except Error as e:
        # TODO Find a way to show errors about referential integrity problems
        return redirect(url_for("update_classroom_page", faculty_id=faculty_id, id=id))
    return redirect(url_for("faculty_detailed", faculty_id=faculty_id))


# course pages#
def courses_page():
    db = current_app.config["db"]
    courses = db.get_all_courses()
    return render_template("courses.html", courses=courses)


def add_course_page():
    form = CourseForm()
    if form.validate_on_submit():
        db = current_app.config['db']
        args = []
        for key, value in form.data.items():
            if key != 'csrf_token':
                args.append(value)
        course = Course(*args)
        db.add_course(course)
        return redirect(url_for('courses_page'))
    return render_template("edit_course.html", form=form, error=None, title="Add Course")


# instructor pages#
def instructors_page():
    db = current_app.config["db"]
    instructors = db.get_all_instructors()
    return render_template("instructors.html", instructors=instructors)


def add_instructor_page():
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
            if isinstance(e, errors.UniqueViolation):
                return render_template("edit_instructor.html", form=form, title="Add Instructor",
                                       error="An instructor with this TR ID already exists")
            if isinstance(e, errors.ForeignKeyViolation):
                return render_template("edit_instructor.html", form=form, title="Add Instructor",
                                       error="No people exists with this TR ID")
            else:
                return render_template("edit_instructor.html", form=form, title="Add Instructor",
                                       error=type(e).__name__ + "-----" + str(e))
        return redirect(url_for("instructors_page"))
    return render_template("edit_instructor.html", form=form, title="Add Instructor", error=None)


def update_instructor_page(id):
    db = current_app.config["db"]
    form = InstructorForm()
    if form.validate_on_submit():
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
        except Error as e:
            if isinstance(e, errors.UniqueViolation):
                return render_template("edit_instructor.html", form=form, title="Update Instructor",
                                       error="An instructor with this TR ID already exists")
            if isinstance(e, errors.ForeignKeyViolation):
                return render_template("edit_instructor.html", form=form, title="Update Instructor",
                                       error="No people exists with this TR ID")
            else:
                return render_template("edit_instructor.html", form=form, title="Update Instructor",
                                       error=type(e).__name__ + "-----" + str(e))
        return redirect(url_for("instructors_page"))
    instructor = db.get_instructor(id)
    form.tr_id.data = instructor.tr_id
    form.room_id.data = instructor.room_id
    form.doctorates.data = instructor.doctorates
    form.masters.data = instructor.masters
    form.bachelors.data = instructor.bachelors
    form.specialization.data = instructor.specialization
    form.department_id.data = instructor.department_id
    form.faculty_id.data = instructor.faculty_id
    return render_template("edit_instructor.html", form=form, title="Update Instructor", error=None)


def delete_instructor(id):
    db = current_app.config['db']
    db.delete_instructor(id)
    # TODO:https://stackoverflow.com/questions/18290142/multiple-forms-in-a-single-page-using-flask-and-wtforms
    #   fix referential integrity problems
    return redirect(url_for("instructors_page"))


def test_page():
    return render_template("test.html")


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


    else:
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

