Parts Implemented by Cihat Akkiraz
================================

Campus, Faculty and Departments tables that are implemented by this team member will be explained here.

Campus
------

.. figure:: ../../images/cihat_erd.png
    :alt: ERD of Akkiraz
    :align: center

.. code-block:: sql
    :linenos:
    :caption: SQL of Campus Table

    CREATE TABLE IF NOT EXISTS CAMPUS(
        id		            SERIAL 		NOT NULL,
        name 		        VARCHAR(50)	NOT NULL,
        address 	        VARCHAR(80)	NOT NULL,
        city 		        VARCHAR(25),
        size 		        INT,
        foundation_date     DATE,
        phone_number        CHAR(11),   
        campus_image_extension VARCHAR(10) DEFAULT('NO_IMAGE'),
        campus_image_data   bytea, 
        PRIMARY KEY(id)
    );

.. code-block:: sql
    :linenos:
    :caption: SQL of Faculty Table

    CREATE TABLE IF NOT EXISTS FACULTY(
        id				    SERIAL 		NOT NULL,
        campus_id           INT         NOT NULL,
        name 				VARCHAR(100) NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        address 			VARCHAR(80),
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS(id)
    );

.. code-block:: sql
    :linenos:
    :caption: SQL of Department Table

    CREATE TABLE IF NOT EXISTS DEPARTMENT(
        id				    SERIAL 		NOT NULL,
        faculty_id			INT			NOT NULL,
        name 				VARCHAR(100) NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        block_number 		CHAR(1),
        budget			 	INT,
        foundation_date 	DATE,
        phone_number		CHAR(11),
        PRIMARY KEY(id),
        FOREIGN KEY(faculty_id) REFERENCES FACULTY(id)
    );

Classes
-------

Classes are python objects for the user types that ITU DataBees uses.

Classes implemented by this user can be examined in detailed with the given source code from various files.

Classes campus, faculty and department are created and implemented by this user and their corresponding codes are given below.

Campus
++++++

.. code-block:: python
    :linenos:
    :caption: Course class from ``campus.py``

    class Campus:
        def __init__(self, campus_id, name, address, city, size, foundation_date, phone_number, image_extension, image_data):
            self.id = campus_id
            self.name = name
            self.address = address
            self.city = city
            self.size = size
            self.foundation_date = foundation_date
            self.phone_number = phone_number
            self.img_extension = image_extension
            self.img_data = image_data

        def get_campus_id(self):
            return self.id

Faculty
++++++

.. code-block:: python
    :linenos:
    :caption: Faculty class from ``faculty.py``

    class Faculty:
        def __init__(self, faculty_id, campus_id, name, shortened_name, adress, foundation_date, phone_number):
            self.id = faculty_id
            self.campus_id = campus_id
            self.name = name
            self.shortened_name = shortened_name
            self.address = adress
            self.foundation_date = foundation_date
            self.phone_number = phone_number

        def get_faculty_id(self):
            return self.id

Department
++++++

.. code-block:: python
    :linenos:
    :caption: Department class from ``department.py``

    class Department:
        def __init__(self, department_id, faculty_id, name, shortened_name, block_number, budget, foundation_date, phone_number):
            self.id = department_id
            self.faculty_id = faculty_id
            self.name = name
            self.shortened_name = shortened_name
            self.block_number = block_number
            self.budget = budget
            self.foundation_date = foundation_date
            self.phone_number = phone_number

        def get_department_id(self):
            return self.id


View Models
-----------

View models handle GET/POST requests and render pages accordingly.

Models implemented by this user can be examined in detailed with the given source code from ``views.py`` file.

Errors from SQL quarries are handled and required information is shown to user.

Given code snippets below are written by this member.

Campus
+++++++++

.. code-block:: python
    :linenos:
    :caption: View for the Add Campus page

    @login_required
    def campus():
        if(current_user.is_admin):
            db = current_app.config["db"]
            campuses = db.get_campuses()
            campus = {}
            form = add_campus_form()
            error = ''
            if request.method == "POST" and 'delete_campus_flag' in request.form:
                campus_id = request.form['delete_campus_flag']

                try:
                    db.delete_campus(campus_id)
                    return redirect(url_for('campus'))
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
                    if isinstance(e, errors.ForeignKeyViolation):
                        str_e = str(e)
                        if 'faculty' in str_e:
                            error = "There are faculties in this campus! It can not be deleted!"
                    pass
                context = {
                    # 'form': form,
                    'campuses': campuses,
                    'form': form,
                    'error': error
                }
                return render_template('/campuses/campus.html', context=context)
            elif request.method == "POST" and 'add_campus_form' in request.form:
                if(form.validate()):
                    image = request.files['image']
                    filename = secure_filename(image.filename)
                    file_extension = filename.split(".")[-1]
                    filename = filename.split(".")[0]
                    if(validate_image(file_extension)):
                        img_data = request.files['image'].read()
                    else:
                        filename = ""
                        file_extension = "NO_IMAGE"
                        img_data = b''
                    campus = Campus(0, form.name.data, form.address.data, form.city.data, form.size.data,
                                    form.foundation_date.data, form.phone_number.data, file_extension, img_data)
                    try:
                        db.add_campus(campus)
                        return redirect(url_for('campus'))
                    except Error as e:
                        error = tidy_error(e)
                    print(error)
                    return redirect(url_for('campus'))
                else:
                    error = form.errors
                    context = {
                        # 'form': form,
                        'campuses': campuses,
                        'form': form,
                        'error':error
                    }
                    return render_template('/campuses/campus.html', context=context)
            elif request.method == "POST" and "redirect_edit_page" in request.form:
                campus_form_id = request.form['redirect_edit_page']
                return redirect(url_for('campus_detailed', campus_id=campus_form_id))
            context = {
                # 'form': form,
                'campuses': campuses,
                'form': form,
                'error':error
            }
            return render_template('/campuses/campus.html', context=context)

.. code-block:: python
    :linenos:
    :caption: View for the Campus Edit page

    @login_required
    def campus_detailed(campus_id):
        if(current_user.is_admin):
            db = current_app.config["db"]
            campus = db.get_campus(campus_id)
            edit_campus_form = add_campus_form()
            add_faculty = add_faculty_form()
            if(campus.img_data is None):
                image = ""
                image_extension = ""
            elif(campus.img_extension != "NO_IMAGE"):
                image = b64encode(campus.img_data)
                image = image.decode('utf-8')
                image_extension = campus.img_extension
            else:
                image = ""
                image_extension = ""
            faculties = db.get_faculties_from_campus(campus.id)

            if request.method == "POST" and 'change_picture' in request.form:
                image = request.files['image']
                filename = secure_filename(image.filename)
                file_extension = filename.split(".")[-1]
                filename = filename.split(".")[0]
                img_data = b''
                error=''
                if(validate_image(file_extension)):
                    img_data = request.files['image'].read()
                updated_campus = Campus(campus_id, campus.name, campus.address, campus.city, campus.size,
                                        campus.foundation_date, campus.phone_number, file_extension, img_data)
                try:
                    db.update_campus(updated_campus)
                    return redirect(url_for('campus_detailed', campus_id=campus_id))
                except Error as e:
                    error = tidy_error(e)
                    pass
                context = {
                    'Campus': campus,
                    'edit_campus_form': edit_campus_form,
                    'campus_image': image,
                    'campus_image_extension': image_extension,
                    'add_faculty_form': add_faculty,
                    'faculties': faculties,
                    'image_added': True,
                    'error': error
                }
                return render_template('/campuses/campus_detailed.html', context=context)
            elif request.method == "POST" and "delete_image" in request.form:
                file_extension = ""
                img_data = b""
                updated_campus = Campus(campus_id, campus.name, campus.address, campus.city, campus.size,
                                        campus.foundation_date, campus.phone_number, file_extension, img_data)
                error =''
                try:
                    db.update_campus(updated_campus)
                    return redirect(url_for('campus_detailed', campus_id=campus_id))
                except Error as e:
                    error = tidy_error(e)
                    pass
                context = {
                    'Campus': campus,
                    'edit_campus_form': edit_campus_form,
                    'campus_image': image,
                    'campus_image_extension': image_extension,
                    'add_faculty_form': add_faculty,
                    'faculties': faculties,
                    'image_added': True,
                    'error': error
                }
                return render_template('/campuses/campus_detailed.html', context=context)
            elif request.method == "POST" and 'add_faculty_form' in request.form:
                error = ''
                if(add_faculty.validate()):
                    faculty = Faculty(0, request.form['add_faculty_form'], add_faculty.name.data, add_faculty.shortened_name.data,
                                    add_faculty.address.data, add_faculty.foundation_date.data, add_faculty.phone_number.data)
                    try:
                        db.add_faculty(faculty)
                        return redirect(url_for('campus_detailed', campus_id=campus.id))
                    except Error as e:
                        error = tidy_error(e)
                        pass
                    
                context = {
                        'Campus': campus,
                        'edit_campus_form': edit_campus_form,
                        'campus_image': image,
                        'campus_image_extension': image_extension,
                        'add_faculty_form': add_faculty,
                        'faculties': faculties,
                        'image_added': True,
                        'error': error
                }
                return render_template('/campuses/campus_detailed.html', context=context)
            elif request.method == "POST" and 'edit_campus_form' in request.form:
                campus_id = campus.id
                updated_campus = Campus(campus_id, edit_campus_form.name.data, edit_campus_form.address.data, edit_campus_form.city.data, edit_campus_form.size.data,
                                        edit_campus_form.foundation_date.data, edit_campus_form.phone_number.data, campus.img_extension, campus.img_data)
                error =''
                if(edit_campus_form.validate()):
                    try:
                        db.update_campus(updated_campus)
                        return redirect(url_for('campus_detailed', campus_id=campus.id))
                    except Error as e:
                        error = tidy_error(e)
                        pass
                    add_faculty = add_faculty_form()
                    edit_campus_form = add_campus_form()
                else:
                    error = edit_campus_form.errors
                context = {
                    'Campus': campus,
                    'edit_campus_form': edit_campus_form,
                    'campus_image': image,
                    'campus_image_extension': image_extension,
                    'add_faculty_form': add_faculty,
                    'faculties': faculties,
                    'image_added': True,
                    'error': error,
                    'update_error': error
                }
                return render_template('/campuses/campus_detailed.html', context=context)
            elif request.method == "POST" and 'delete_faculty_flag' in request.form:
                faculty_delete_id = request.form['delete_faculty_flag']
                error =''
                try:
                    db.delete_faculty(faculty_delete_id)
                    return redirect(url_for('campus_detailed', campus_id=campus.id))
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
                    if isinstance(e, errors.ForeignKeyViolation):
                        str_e = str(e)
                        if 'department' in str_e:
                            remove_error = "There are departments in this faculty! It can not be deleted!"
                    pass
                context = {
                    'Campus': campus,
                    'edit_campus_form': edit_campus_form,
                    'campus_image': image,
                    'campus_image_extension': image_extension,
                    'add_faculty_form': add_faculty,
                    'faculties': faculties,
                    'image_added': True,
                    'remove_error': remove_error
                }
                return render_template('/campuses/campus_detailed.html', context=context)
            elif request.method == "POST" and 'redirect_edit_page' in request.form:
                faculty_form_id = request.form['redirect_edit_page']
                return redirect(url_for('faculty_detailed', faculty_id=faculty_form_id))

            context = {
                # 'add_faculty_form': add_facultyForm,
                'Campus': campus,
                'edit_campus_form': edit_campus_form,
                'campus_image': image,
                'campus_image_extension': image_extension,
                'add_faculty_form': add_faculty,
                'faculties': faculties,
                'image_added': True,
            }
            return render_template('/campuses/campus_detailed.html', context=context)

.. code-block:: python
    :linenos:
    :caption: View for the Upload Campus image

    @login_required
    def upload_campus_image(request):
        form = upload_campus_image_form()
        if request.method == 'POST':
            imagefile = flask.request.files.get('imagefile', '')
            if 'imagefile' not in request.files:
                flash('No file part')
                return redirect(request.url)
            # if user does not select file, browser also
            # submit an empty part without filename
            if imagefile.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if imagefile and allowed_file(imagefile.filename):
                filename = secure_filename(imagefile.filename)
                imagefile.save(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('list_campus',
                                        filename=filename))
        return form

Faculty
+++++++++

.. code-block:: python
    :linenos:
    :caption: View for the Faculty Edit page

    @login_required
    def faculty_detailed(faculty_id):
        if current_user.is_admin:
            db = current_app.config["db"]
            classrooms = db.get_all_classrooms_by_faculty(faculty_id)
            faculty = db.get_faculty(faculty_id)
            edit_faculty_form = add_faculty_form()
            add_department = add_department_form()
            departments = db.get_departments_from_faculty(faculty_id)
            context = {
                'Faculty': faculty,
                'edit_faculty_form': edit_faculty_form,
                'add_department_form': add_department,
                'departments': departments,
            }
            if request.method == "POST" and 'add_department_form' in request.form:
                if(add_department.validate()):
                    department = Department(0, faculty_id, add_department.name.data, add_department.shortened_name.data, add_department.block_number.data,
                                            add_department.budget.data, add_department.foundation_date.data, add_department.phone_number.data)
                    try:
                        db.add_department(department)
                        return redirect(url_for('faculty_detailed', faculty_id=faculty.id, classrooms=classrooms))
                    except Error as e:
                        add_error = tidy_error(e)
                        pass
                    context['add_error'] = add_error
                return render_template('/campuses/faculty_detailed.html', context=context, classrooms=classrooms)
            elif request.method == "POST" and 'edit_faculty_form' in request.form:
                if(edit_faculty_form.validate()):
                    updated_faculty = Faculty(faculty_id, faculty.campus_id, edit_faculty_form.name.data, edit_faculty_form.shortened_name.data,
                                            edit_faculty_form.address.data, edit_faculty_form.foundation_date.data, edit_faculty_form.phone_number.data)
                    try:
                        db.update_faculty(updated_faculty)
                        return redirect(url_for('faculty_detailed', faculty_id=faculty.id, classrooms=classrooms))
                    except Error as e:
                        update_error = tidy_error(e)
                        pass
                    context['update_error'] = update_error
                    context['faculty'] = updated_faculty
                else:
                    print(edit_faculty_form.errors)
                return render_template('/campuses/faculty_detailed.html', context=context, classrooms=classrooms)
            elif request.method == "POST" and 'delete_department_flag' in request.form:
                try:
                    db.delete_department(request.form['delete_department_flag'])
                    return redirect(url_for('faculty_detailed', faculty_id=faculty.id, classrooms=classrooms))
                except Error as e:
                    remove_error = type(e).__name__ + '----' + str(e)
                    if('student' in remove_error):
                        remove_error = "The department cannot be deleted because of registered students"
                    pass
                context['remove_error'] = remove_error
                return render_template('/campuses/faculty_detailed.html', context=context, classrooms=classrooms)
            elif request.method == "POST" and 'redirect_edit_page' in request.form:
                department_form_id = request.form['redirect_edit_page']
                return redirect(url_for('department_detailed', department_id=department_form_id))

            return render_template('/campuses/faculty_detailed.html', context=context, classrooms=classrooms)


Department
+++++++++

.. code-block:: python
    :linenos:
    :caption: View for the Department Edit page

    @login_required
    def department_detailed(department_id):
        if current_user.is_admin:
            db = current_app.config["db"]
            department = db.get_department(department_id)
            edit_department_form = add_department_form()
            context = {
                'Department': department,
                'edit_department_form': edit_department_form
            }
            if(request.method == "POST" and 'edit_department_form' in request.form):
                if(edit_department_form.validate()):
                    updated_department = Department(department.id, department.faculty_id, edit_department_form.name.data, edit_department_form.shortened_name.data,
                                                    edit_department_form.block_number.data, edit_department_form.budget.data, edit_department_form.foundation_date.data, edit_department_form.phone_number.data)
                    try:
                        db.update_department(updated_department)
                        return redirect(url_for('department_detailed', department_id=department.id))
                    except Error as e:
                        error = type(e).__name__ + '----' + str(e)
                        pass
                    return redirect(url_for('department_detailed', department_id=department.id))
                else:
                    context['error']=edit_department_form.errors
                    
            return render_template('/campuses/department_detailed.html', context=context)

Database Queries
----------------

Database queries are handled via ``database.py`` file by constructing a Database class and using ``psycopg2`` library as the PostgreSQL driver.

Below are the related class methods implemented by this member:

Campus
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Campus Table

    def add_campus(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into campus (name, address, city, size, foundation_date, phone_number,campus_image_extension,campus_image_data) values (%s, %s, %s, %s, %s,%s,%s ,%s)"
            cursor.execute(query, (campus.name, campus.address,
                                   campus.city, campus.size, campus.foundation_date,
                                   campus.phone_number,campus.img_extension,
                                   campus.img_data))
            connection.commit()
        print('End of the campus add function')
        self.campuses[campus.id] = campus
        return campus.id

    def delete_campus(self, campus_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from campus where (id = %s)"
            cursor.execute(query, (campus_id,))
            connection.commit

    def update_campus(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            print('hey')
            cursor = connection.cursor()
            query = "update campus set name = %s, address = %s, city = %s, size = %s, foundation_date = %s, phone_number = %s,campus_image_extension = %s, campus_image_data = %s where (id= %s)"
            cursor.execute(query, (campus.name, campus.address, campus.city,
                                   campus.size, campus.foundation_date, campus.phone_number, campus.img_extension, campus.img_data,
                                   campus.id))
            connection.commit

    def update_campus_image(self, campus):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update campus set file_extension = %s, image_data = %s where (id= % s)"
            cursor.execute(query, (campus.file_extension, campus.image_data, campus.id))
            connection.commit

    def get_campuses(self):
        campuses = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from campus order by id asc"
            cursor.execute(query)
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                campus = Campus(*row[:])
                campuses.append((campus.id, campus))
        return campuses

    def get_campus(self, campus_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from campus where (id = %s)"
            cursor.execute(query, (campus_id,))
            if (cursor.rowcount == 0):
                return None
        campus_ = Campus(*cursor.fetchone()[:])  # Inline unpacking of a tuple
        return campus_

Faculty
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Faculty Table

    def add_faculty(self, faculty):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into faculty (campus_id, name, shortened_name, address, foundation_date, phone_number) values (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (faculty.campus_id, faculty.name, faculty.shortened_name,
                                   faculty.address, faculty.foundation_date, faculty.phone_number))
            connection.commit

    def update_faculty(self, faculty):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update faculty set name = %s, shortened_name = %s, address = %s, foundation_date = %s, phone_number = %s where (id = %s)"
            cursor.execute(query, (faculty.name, faculty.shortened_name, faculty.address,
                                   faculty.foundation_date, faculty.phone_number, faculty.id))
            connection.commit

    def delete_faculty(self, faculty_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from faculty where (id = %s)"
            cursor.execute(query, (faculty_id,))
            connection.commit

    def get_faculties_from_campus(self, campus_id):
        faculties = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from faculty where (campus_id = %s) order by id asc"
            cursor.execute(query, (campus_id,))
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                faculty = Faculty(*row[:])
                faculties.append((faculty.id, faculty))
        return faculties

    def get_faculty(self, faculty_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from faculty where (id = %s)"
            cursor.execute(query, (faculty_id,))
            if (cursor.rowcount == 0):
                return None
        # Inline unpacking of a tuple
        faculty_ = Faculty(*cursor.fetchone()[:])
        return faculty_

Department
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Department Table

    def add_department(self, department):
        print('Enter add department')
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            print(department)
            query = "insert into department (faculty_id, name, shortened_name, block_number, budget, foundation_date, phone_number) values (%s, %s, %s,%s,%s,%s,%s)"
            cursor.execute(query,
                           (department.faculty_id, department.name, department.shortened_name,
                            department.block_number, department.budget, department.foundation_date,
                            department.phone_number))
            connection.commit

    def update_department(self, department):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update department set name = %s, shortened_name = %s, block_number = %s, budget = %s, foundation_date = %s, phone_number = %s where (id = %s)"
            cursor.execute(query, (
                department.name, department.shortened_name, department.block_number,
                department.budget,
                department.foundation_date, department.phone_number, department.id,))
            connection.commit

    def delete_department(self, department_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from department where (id = %s)"
            cursor.execute(query, (department_id,))
            connection.commit

    def get_departments_from_faculty(self, faculty_id):
        departments = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from department where (faculty_id = %s) order by id asc"
            cursor.execute(query, (faculty_id,))
            print('Cursor.rowcount', cursor.rowcount)
            for row in cursor:
                department = Department(*row[:])
                departments.append((department.id, department))
        return departments

    def get_department(self, department_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from department where (id = %s)"
            cursor.execute(query, (department_id,))
            if (cursor.rowcount == 0):
                return None
        # Inline unpacking of a tuple
        department_ = Department(*cursor.fetchone()[:])
        return department_


Templates
---------

Following templates are written by **this user**:
    - ``campuses/campus.html``
    - ``campuses/campus_detailed.html``
    - ``campuses/faculty_detailed.html``
    - ``campuses/department_detailed.html.html``

Following templates are written both by **this member** and **other teammates**:
    - ``layout.html``
    - ``login.html``