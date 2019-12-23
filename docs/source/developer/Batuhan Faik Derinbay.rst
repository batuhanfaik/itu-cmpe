Parts Implemented by Batuhan Faik Derinbay
==========================================

Tables
------

Tables implemented by this user can be examined in detailed with the given source code from the ``dbinit.py`` file.

Tables People, Assistant and Student are created and implemented by this user and their corresponding statements are given below.

.. code-block:: sql
    :linenos:
    :caption: SQL of People Table

    CREATE TABLE IF NOT EXISTS PEOPLE (
            tr_id BIGINT PRIMARY KEY NOT NULL,
            name VARCHAR(40) NOT NULL,
            surname VARCHAR(40) NOT NULL,
            phone_number VARCHAR(20) not null,
            email VARCHAR(60) NOT NULL,
            pass varchar(256) not null,
            person_category SMALLINT NOT NULL,
            mother_fname varchar(40) null,
            father_fname varchar(40) null,
            gender char(1) null,
            birth_city varchar(50) null,
            birth_date date not null,
            birth_date varchar(50) not null,
            id_reg_district varchar(50) not null,
            photo_name varchar(256),
            photo_extension varchar(10),
            photo_data bytea,
            unique (tr_id),
            unique (email)
        );

.. code-block:: sql
    :linenos:
    :caption: SQL of Assistant Table

    CREATE TABLE IF NOT EXISTS ASSISTANT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id) on delete cascade on update cascade NOT NULL,
        faculty_id int references FACULTY(id) on delete cascade on update cascade not null,
        supervisor bigint references INSTRUCTOR(tr_id) on delete cascade on update cascade not null,
        assistant_id bigint not null,
        bachelors varchar(80) not null,
        degree varchar(80) not null,
        grad_gpa real not null,
        research_area varchar(100) not null,
        office_day varchar(9) null default 'none',
        office_hour_start time null,
        office_hour_end time null,
        unique (assistant_id)
    );

.. code-block:: sql
    :linenos:
    :caption: SQL of Student Table

    CREATE domain credit as real check (
        ((value >= 0) and (value <=250))
    );

    CREATE TABLE IF NOT EXISTS STUDENT (
        tr_id BIGINT PRIMARY KEY references PEOPLE(tr_id)  on delete cascade on update cascade NOT NULL,
        faculty_id int references FACULTY(id) on delete cascade on update cascade not null,
        department_id int references DEPARTMENT(id) on delete cascade on update cascade not null,
        student_id bigint not null,
        semester smallint not null default 1,
        grade smallint not null default 1,
        gpa real not null default 0,
        credits_taken credit not null,
        minor boolean not null default false,
        unique (student_id)
    );

Inserting Data
++++++++++++++

Filling tables of ITU DataBees with initialization data for easier demonstration of the functions.

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into People Table

    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (1,'fadmin', 'fatih', '1', 
    'fadmin@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (2,'badmin', 'batu', '1', 
    'badmin@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (3,'cadmin', 'cihat', '1', 
    'cadmin@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (4,'zadmin', 'zeynep', '1', 
    'zadmin@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '0', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11,'finstructor', 'fatih', '1', 
    'finstructor@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22,'binstructor', 'batu', '1', 
    'binstructor@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33,'cinstructor', 'cihat', '1', 
    'cinstructor@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44,'zinstructor', 'zeynep', '1', 
    'zinstructor@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '2', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (111,'fstudent', 'fatih', '1', 
    'fstudent@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (222,'bstudent', 'batu', '1', 
    'bstudent@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (333,'cstudent', 'cihat', '1', 
    'cstudent@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (444,'zstudent', 'zeynep', '1', 
    'zstudent@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '5', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (11111,'fassistant', 'fatih', '1', 
    'fassistant@itu.edu.tr','$pbkdf2-sha256$29000$pPQ.RwgB4Nxbq7V2DmGM8Q$4lFUXxu17es8iNJHSD/w/FM6Y/5JaF7bvekDxhRmAeU',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (22222,'bassistant', 'batu', '1', 
    'bassistant@itu.edu.tr','$pbkdf2-sha256$29000$cc557907RyiFEOK813ovJQ$Xnrg4Tfl5QqpZoeVfHmBaA4A./ZK.6obUc2WXNIIu3g',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (33333,'cassistant', 'cihat', '1', 
    'cassistant@itu.edu.tr','$pbkdf2-sha256$29000$PMeYc865d641BiBE6N2b8w$BE4L4t9zfdrZvKYuJRX0/EnpkiSA2n/TAIXwfmhTj1c',
    '3', '2019-10-10', 'a','b');
    insert into people (tr_id, name, surname, phone_number, email, pass, person_category,
    birth_date, id_reg_city, id_reg_district) values (44444,'zassistant', 'zeynep', '1', 
    'zassistant@itu.edu.tr','$pbkdf2-sha256$29000$3RsjZAxByLm3ViqF8F7rXQ$HkPwZXe73FrvDuVJQ3JC1ExmmcIvbAwpbnhzhMmqa0w',
    '3', '2019-10-10', 'a','b');

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into Assistant Table

    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (11111, 1, 11, 500180707, 'ITU', 'Computer Engineering', 3.7, 'Machine Learning', 'Monday', '10:30', '12:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (22222, 2, 22, 500180704, 'ITU', 'Computer Engineering', 3.8, 'Machine Learning', 'Tuesday', '12:30', '14:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (33333, 3, 33, 500180705, 'ITU', 'Computer Engineering', 3.9, 'Machine Learning', 'Wednesday', '14:30', '16:30');
    insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values 
    (44444, 4, 44, 500150150, 'ITU', 'Computer Engineering', 4.0, 'Machine Learning', 'Thursday', '16:30', '18:30');

.. code-block:: sql
    :linenos:
    :caption: SQL for Inserting Data into Student Table

    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (111, 1, 1, 150180707, 69.5, 0);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (222, 1, 2, 150180704, 200, 4);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (333, 2, 1, 150180705, 200, 4);
    insert into student (tr_id, faculty_id, department_id, student_id, credits_taken, gpa) values 
    (444, 3, 1, 150150150, 200, 4);

Re-Initialization of DataBees
+++++++++++++++++++++++++++++

Code below removes clears all content of the database.

.. code-block:: sql
    :linenos:
    :caption: SQL for Resetting the Database

    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;

Classes
-------

Classes are python objects for the user types that ITU DataBees uses.

Classes implemented by this user can be examined in detailed with the given source code from various files.

Classes person, assistant and student are created and implemented by this user and their corresponding codes are given below.

Person
++++++

.. code-block:: python
    :linenos:
    :caption: Person class from ``person.py``

    class Person(UserMixin):
        def __init__(self, tr_id, name, surname, phone_number, email, password, person_category,
                     mother_fname, father_fname, gender, birth_city, birth_date, id_reg_city,
                     id_reg_district, photo_name, photo_extension, photo_data):
            self.tr_id = tr_id
            self.id = email
            self.name = name
            self.surname = surname
            self.phone_number = phone_number
            self.email = email
            self.password = password
            self.person_category = person_category
            self.mother_fname = mother_fname
            self.father_fname = father_fname
            self.gender = gender
            self.birth_city = birth_city
            self.birth_date = birth_date
            self.id_reg_city = id_reg_city
            self.id_reg_district = id_reg_district
            self.photo_name = photo_name
            self.photo_extension = photo_extension
            self.photo_data = photo_data

            if self.photo_data:
                self.photo = b64encode(self.photo_data)
                self.photo = self.photo.decode('utf-8')

            self.active = True
            if self.person_category == 0:
                self.is_admin = True
            else:
                self.is_admin = False

            @property
            def get_id(self):
                return self.email

            @property
            def is_active(self):
                return self.active

Assistant
+++++++++

.. code-block:: python
    :linenos:
    :caption: Assistant class from ``assistant.py``

    class Assistant(UserMixin):
    def __init__(self, tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa,
                 research_area, office_day, office_hour_start, office_hour_end):
        self.tr_id = tr_id
        self.faculty_id = faculty_id
        self.supervisor = supervisor
        self.assistant_id = assistant_id
        self.bachelors = bachelors
        self.degree = degree
        self.grad_gpa = grad_gpa
        self.research_area = research_area
        self.office_day = office_day
        self.office_hour_start = office_hour_start
        self.office_hour_end = office_hour_end

Student
+++++++

.. code-block:: python
    :linenos:
    :caption: Student class from ``student.py``

    class Student(UserMixin):
        def __init__(self, tr_id, faculty_id, department_id, student_id, semester, grade, gpa,
                     credits_taken, minor):
            self.tr_id = tr_id
            self.faculty_id = faculty_id
            self.department_id = department_id
            self.student_id = student_id
            self.semester = semester
            self.grade = grade
            self.gpa = gpa
            self.credits_taken = credits_taken
            self.minor = minor

View Models
-----------

View models handle GET/POST requests and render pages accordingly.

Models implemented by this user can be examined in detailed with the given source code from ``views.py`` file.

Given code snippets below are written by this user.

People
++++++

.. code-block:: python
    :linenos:
    :caption: Model for the people page

    @login_required
    def people_page():
        db = current_app.config["db"]
        try:
            people = db.get_people()
        except Error as e:
            error = tidy_error(e)
            return render_template("people.html", people=None, values={}, error=error)
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
            try:
                db.add_person(person)
                people = db.get_people()
                return render_template("people.html", people=sorted(people), values={}, error=None)
            except Error as e:
                error = tidy_error(e)
                people = db.get_people()
                return render_template("people.html", people=sorted(people), values={}, error=error)

.. code-block:: python
    :linenos:
    :caption: Model for the person page

    @login_required
    def person_page(tr_id):
        db = current_app.config["db"]
        try:
            person = db.get_person(tr_id)
        except Error as e:
            error = tidy_error(e)
            people = db.get_people()
            return render_template("people.html", people=sorted(people), values={})
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
                try:
                    db.update_person(person, tr_id)
                    return redirect(url_for("person_page", tr_id=person.tr_id, error=None))
                except Error as e:
                    error = tidy_error(e)
                    person = db.get_person(tr_id)
                    return render_template("person.html", tr_id=tr_id, error=error, person=person)
            elif request.form["update_button"] == "delete":
                try:
                    db.delete_person(tr_id)
                    people = db.get_people()
                    return redirect(url_for("people_page", people=sorted(people), values={}))
                except Error as e:
                    error = tidy_error(e)
                    return render_template("person.html", tr_id=tr_id, error=error, person=person)

.. code-block:: python
    :linenos:
    :caption: Form handling function for the people page

    def validate_people_form(form):
        form.data = {}
        form.errors = {}
        db = current_app.config["db"]

        form_tr_id = form.get("tr_id")
        if not form_tr_id.isdigit():
            form.errors["tr_id"] = "Please enter only numeric characters!"
            return len(form.errors) == 0

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

Assistant
+++++++++

.. code-block:: python
    :linenos:
    :caption: Model for the assistants page

    @login_required
    def assistants_page():
        db = current_app.config["db"]
        try:
            assistants = db.get_assistants()
        except Error as e:
            error = tidy_error(e)
            return render_template("assistants.html", assistants=None, values={}, error=error)
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
            try:
                db.add_assistant(assistant)
                assistants = db.get_assistants()
                return render_template("assistants.html", assistants=sorted(assistants), values={},
                                       error=None)
            except Error as e:
                error = tidy_error(e)
                return render_template("assistants.html", assistants=sorted(assistants), values={},
                                       error=error)

.. code-block:: python
    :linenos:
    :caption: Model for the assistant page

    @login_required
    def assistant_page(tr_id):
        db = current_app.config["db"]
        try:
            assistant = db.get_assistant(tr_id)
        except Error as e:
            error = tidy_error(e)
            assistants = db.get_assistants()
            return redirect(url_for("assistants_page", assistants=sorted(assistants), values={},
                                    error=error))
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
                form_supervisor = request.form["supervisor"]
                form_assistant_id = request.form["assistant_id"]
                form_bachelors = request.form["bachelors"]
                form_degree = request.form["degree"]
                form_grad_gpa = request.form["grad_gpa"]
                form_research_area = request.form["research_area"]
                form_office_day = request.form["office_day"]
                form_office_hour_start = request.form["office_hour_start"]
                form_office_hour_end = request.form["office_hour_end"]

                assistant = Assistant(form_tr_id, form_faculty_id, form_supervisor, form_assistant_id,
                                      form_bachelors, form_degree, form_grad_gpa, form_research_area,
                                      form_office_day, form_office_hour_start, form_office_hour_end)
                db = current_app.config["db"]
                try:
                    db.update_assistant(assistant, tr_id)
                    return redirect(url_for("assistant_page", tr_id=assistant.tr_id, error=None))
                except Error as e:
                    error = tidy_error(e)
                    assistant = db.get_assistant(tr_id)
                    return render_template("assistant.html", tr_id=tr_id, error=error,
                                           assistant=assistant)
            elif request.form["update_button"] == "delete":
                try:
                    db.delete_assistant(tr_id)
                    assistants = db.get_assistants()
                    return redirect(url_for("assistants_page", assistants=sorted(assistants), values={},
                                            error=None))
                except Error as e:
                    error = tidy_error(e)
                    assistant = db.get_assistant(tr_id)
                    return render_template("assistant.html", tr_id=tr_id, error=error,
                                           assistant=assistant)

.. code-block:: python
    :linenos:
    :caption: Form handling function for the assistants page

    def validate_assistants_form(form):
        form.data = {}
        form.errors = {}
        db = current_app.config["db"]

        form_tr_id = form.get("tr_id")
        if not form_tr_id.isdigit():
            form.errors["tr_id"] = "Please enter only numeric characters!"
            return len(form.errors) == 0
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

Student
+++++++

.. code-block:: python
    :linenos:
    :caption: Model for the students page

    @login_required
    def students_page():
        db = current_app.config["db"]
        try:
            students = db.get_students()
        except Error as e:
            error = tidy_error(e)
            return render_template("students.html", students=None, values={}, error=error)
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
            try:
                db.add_student(student)
                students = db.get_students()
                return render_template("students.html", students=sorted(students), values={},
                                       error=None)
            except Error as e:
                error = tidy_error(e)
                return render_template("students.html", students=sorted(students), values={},
                                       error=error)

.. code-block:: python
    :linenos:
    :caption: Model for the student page

    @login_required
    def student_page(tr_id):
        db = current_app.config["db"]
        try:
            student = db.get_student(tr_id)
        except Error as e:
            error = tidy_error(e)
            students = db.get_students()
            return redirect(
                url_for("students_page", students=sorted(students), values={}, error=error))
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
                try:
                    db.update_student(student, tr_id)
                    return redirect(url_for("student_page", tr_id=student.tr_id, error=None))
                except Error as e:
                    error = tidy_error(e)
                    student = db.get_student(tr_id)
                    return render_template("student.html", tr_id=tr_id, error=error, student=student)
            elif request.form["update_button"] == "delete":
                try:
                    db.delete_student(tr_id)
                    students = db.get_students()
                    return redirect(
                        url_for("students_page", students=sorted(students), values={}, error=None))
                except Error as e:
                    error = tidy_error(e)
                    student = db.get_student(tr_id)
                    return render_template("student.html", tr_id=tr_id, error=error, student=student)

.. code-block:: python
    :linenos:
    :caption: Form handling function for the students page

    def validate_students_form(form):
        form.data = {}
        form.errors = {}
        db = current_app.config["db"]

        form_tr_id = form.get("tr_id")
        if not form_tr_id.isdigit():
            form.errors["tr_id"] = "Please enter only numeric characters!"
            return len(form.errors) == 0
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

Others
++++++

.. code-block:: python
    :linenos:
    :caption: Function for simplifying error messages

    def tidy_error(error):
        error_ = type(error).__name__ + '----' + str(error)
        try:
            error_ = re.findall(r":\s{1}([A-z\s\W\d]*)", error_)[0]
        except IndexError:
            error_ = None
        if error_ is None:
            error_ = "There was an error completing your request. Please try again later!" + "\n" + type(
                error).__name__ + "\n" + str(error)
        return error_

.. code-block:: python
    :linenos:
    :caption: Model for the landing page

    def landing_page():
        return render_template("index.html")


The following code snippet is written together as a team.

.. code-block:: python
    :linenos:
    :caption: Model for the login page

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

Database Queries
----------------

Database queries are handled via ``database.py`` file by constructing a Database class and using ``psycopg2`` library as the PostgreSQL driver.

Below are the related class methods implemented by this user:

Person
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the People table

        def add_person(self, person):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "insert into people (tr_id, name, surname, phone_number, email, pass, " \
                        "person_category, mother_fname, father_fname, gender, birth_city, birth_date, " \
                        "id_reg_city, id_reg_district, photo_name, photo_extension, photo_data) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                       person.email, person.password, person.person_category,
                                       person.mother_fname, person.father_fname, person.gender,
                                       person.birth_city, person.birth_date, person.id_reg_city,
                                       person.id_reg_district, person.photo_name,
                                       person.photo_extension, person.photo_data))
                connection.commit()
            self.people[person.tr_id] = person
            return person.tr_id

        def update_person(self, person, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "update people set tr_id = %s, name = %s, surname = %s, phone_number = %s, " \
                        "email = %s, pass = %s, person_category = %s, mother_fname = %s, " \
                        "father_fname = %s, gender = %s, birth_city = %s, birth_date = %s, " \
                        "id_reg_city = %s, id_reg_district = %s, photo_name = %s, photo_extension = %s, photo_data = %s where (tr_id = %s)"
                cursor.execute(query, (person.tr_id, person.name, person.surname, person.phone_number,
                                       person.email, person.password, person.person_category,
                                       person.mother_fname, person.father_fname, person.gender,
                                       person.birth_city, person.birth_date, person.id_reg_city,
                                       person.id_reg_district, person.photo_name,
                                       person.photo_extension, person.photo_data, tr_id))
                connection.commit

        def delete_person(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "delete from people where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                connection.commit

        def get_person(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from people where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                if (cursor.rowcount == 0):
                    return None
            person_ = Person(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return person_

        def get_person_email(self, email):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from people where (email = %s)"
                cursor.execute(query, (email,))
                if (cursor.rowcount == 0):
                    return None
            person_ = Person(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return person_

        def get_people(self):
            people = []
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from people order by name, surname"
                cursor.execute(query)
                for row in cursor:
                    person = Person(*row[:])
                    people.append((person.tr_id, person))
            return people

Assistant
+++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Assistant table

        def add_assistant(self, assistant):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "insert into assistant (tr_id, faculty_id, supervisor, assistant_id, bachelors, degree, grad_gpa, research_area, office_day, office_hour_start, office_hour_end) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(query, (
                    assistant.tr_id, assistant.faculty_id, assistant.supervisor, assistant.assistant_id,
                    assistant.bachelors, assistant.degree, assistant.grad_gpa, assistant.research_area,
                    assistant.office_day, assistant.office_hour_start, assistant.office_hour_end))
                connection.commit()
            self.assistants[assistant.tr_id] = assistant
            return assistant.tr_id

        def update_assistant(self, assistant, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "update assistant set tr_id = %s, faculty_id = %s, supervisor = %s, assistant_id = %s, " \
                        "bachelors = %s, degree = %s, grad_gpa = %s, research_area = %s, office_day = %s, office_hour_start = %s, " \
                        "office_hour_end = %s where (tr_id = %s)"
                cursor.execute(query, (
                    assistant.tr_id, assistant.faculty_id, assistant.supervisor, assistant.assistant_id,
                    assistant.bachelors, assistant.degree, assistant.grad_gpa, assistant.research_area,
                    assistant.office_day, assistant.office_hour_start, assistant.office_hour_end,
                    tr_id))
                connection.commit

        def delete_assistant(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "delete from assistant where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                connection.commit

        def get_assistant(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from assistant where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                if (cursor.rowcount == 0):
                    return None
            assistant_ = Assistant(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return assistant_

        def get_assistant_via_assistant_id(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from assistant where (assistant_id = %s)"
                cursor.execute(query, (tr_id,))
                if (cursor.rowcount == 0):
                    return None
            assistant_ = Assistant(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return assistant_

        def get_assistants(self):
            assistants = []
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from assistant order by assistant_id"
                cursor.execute(query)
                for row in cursor:
                    assistant = Assistant(*row[:])
                    assistants.append((assistant.tr_id, assistant))
            return assistants

Student
+++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Student table

        def add_student(self, student):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "insert into student (tr_id, faculty_id, department_id, student_id, semester, grade, gpa, credits_taken, minor) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(query, (
                    student.tr_id, student.faculty_id, student.department_id, student.student_id,
                    student.semester, student.grade, student.gpa, student.credits_taken, student.minor))
                connection.commit()
            self.students[student.tr_id] = student
            return student.tr_id

        def update_student(self, student, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "update student set tr_id = %s, faculty_id = %s, department_id = %s, student_id = %s, " \
                        "semester = %s, grade = %s, gpa = %s, credits_taken = %s, " \
                        "minor = %s where (tr_id = %s)"
                cursor.execute(query, (
                    student.tr_id, student.faculty_id, student.department_id, student.student_id,
                    student.semester, student.grade, student.gpa, student.credits_taken, student.minor,
                    tr_id))
                connection.commit

        def delete_student(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "delete from student where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                connection.commit

        def get_student(self, tr_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from student where (tr_id = %s)"
                cursor.execute(query, (tr_id,))
                if (cursor.rowcount == 0):
                    return None
            student_ = Student(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return student_

        def get_student_via_student_id(self, student_id):
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from student where (student_id = %s)"
                cursor.execute(query, (student_id,))
                if (cursor.rowcount == 0):
                    return None
            student_ = Student(*cursor.fetchone()[:])  # Inline unpacking of a tuple
            return student_

        def get_students(self):
            students = []
            with dbapi2.connect(self.dbfile) as connection:
                cursor = connection.cursor()
                query = "select * from student order by student_id"
                cursor.execute(query)
                for row in cursor:
                    student = Student(*row[:])
                    students.append((student.tr_id, student))
            return students

Templates
---------

Following templates are written by **this user**:
    - ``people.html``
    - ``person.html``
    - ``assistants.html``
    - ``assistant.html``
    - ``students.html``
    - ``student.html``

*Please also note that these templates include scripts written in JavaScript for validation and better user interface experience. Such an example is given below:*

.. code-block:: javascript
    :linenos:
    :caption: Validation script of People Page

    (function () {
            'use strict';
            window.addEventListener('load', function () {
                // Get the forms we want to add validation styles to
                var forms = document.getElementsByClassName('needs-validation');
                // Loop over them and prevent submission
                var validation = Array.prototype.filter.call(forms, function (form) {
                    form.addEventListener('submit', function (event) {
                        if (form.checkValidity() === false) {
                            event.preventDefault();
                            event.stopPropagation();
                        }
                        form.classList.add('was-validated');
                    }, false);
                });
            }, false);
        })();

Following templates are written both by **this user** and **other teammates**:
    - ``layout.html``
    - ``index.html``
    - ``login.html``

