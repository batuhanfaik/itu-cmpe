Parts Implemented by Zeynep Yetiştiren
==========================================

Tables
------

Staff, Facility and Staff-Facility tables and their CRUD operations are written by this user. Table definitions and attributes can be seen in  ``dbinit.py`` file. Also, they are shown below.

.. code-block:: sql
    :linenos:
    :caption: Facility Table

    CREATE TABLE IF NOT EXISTS FACILITY(
        id				    SERIAL 		NOT NULL,
        campus_id           SERIAL      NOT NULL,
        name 				VARCHAR(40)	NOT NULL,
        shortened_name 		VARCHAR(6)	NOT NULL,
        number_of_workers	INT,
        size             	INT,
        expenses    		INT,
        PRIMARY KEY(id),
        FOREIGN KEY(campus_id) REFERENCES CAMPUS (id) on delete cascade on update cascade
    );

.. code-block:: sql
    :linenos:
    :caption: Staff Table

    CREATE TABLE IF NOT EXISTS STAFF(
        id              BIGINT not null,
        manager_name    VARCHAR(40) null, 
        absences	    INT null, 
        hire_date      	DATE null,
        authority_lvl   INT null,
        department      VARCHAR(40) null,
        social_sec_no   INT null,
        PRIMARY KEY(id),
        FOREIGN KEY(id) REFERENCES PEOPLE (tr_id) on delete cascade on update cascade
    );
.. code-block:: sql
    :linenos:
    :caption: Staff-Facility Table

    CREATE TABLE IF NOT EXISTS STAFF_FACIL(
        title           VARCHAR(20)     NOT NULL,
        from_date 	    DATE            NOT NULL, 
        to_date 	    DATE, 
        salary  	    INT             NOT NULL, 
        facility_id	    BIGINT          NOT NULL, 
        staff_id        BIGINT          NOT NULL,
        duty         	VARCHAR(20)	    NOT NULL,
        FOREIGN KEY(facility_id) REFERENCES FACILITY (id) on delete cascade on update cascade,
        FOREIGN KEY(staff_id) REFERENCES STAFF (id) on delete cascade on update cascade, 
        PRIMARY KEY(facility_id,staff_id)
    );

Example Data
++++++++++++++

Some examples for these tables are shown below. More examples can be seen in ``dbinit.py`` file.

.. code-block:: sql
    :linenos:
    :caption: Example Data for Staff Table

    insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('1', 'Manager1', '1', '2019-12-12','12345','Finance ','1');
    insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('2', 'Manager2', '0', '2019-12-12','12344','Information Tech','2');
    insert into staff (id,manager_name,absences,hire_date,social_sec_no,department,authority_lvl) values ('3', 'Manager3', '1', '2019-12-12','12345','Information Tech','1');




.. code-block:: sql
    :linenos:
    :caption: Example Data for Facility Table

    insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (1, 1, 'Yemekhane', 'YMK', '50', '1400', '70000');
    insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (2, 2, 'Kütüphane', 'LIB', '50', '1400', '50000');
    insert into facility (id, campus_id, name, shortened_name, number_of_workers, size, expenses) values (3, 4, 'Bilgi Işlem', 'BIDB', '50', '1400', '80000');

.. code-block:: sql
    :linenos:
    :caption: Example Data for Staff-Facility Table

    insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('Manager ','2019-12-12', '2019-12-12', '2000', 1, 44, 'Leads staff');
    insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('Security', '2019-12-12', '2019-12-12', '2000', 2, 4, 'Secure books');
    insert into staff_facil (title, from_date, to_date, salary, facility_id, staff_id, duty) values ('Project Assistant','2019-12-12', '2019-12-12', '2000', 3, 44, 'Help the group');


Classes
-------

Classes are created for table definitions in Python. Classes can be examined in ``staff.py``, ``facility.py`` and ``staff_facil.py`` files.

Classes Staff, Facility and Staff_facil are created and implemented by this user and their are shown below.

Staff
++++++

.. code-block:: python
    :linenos:
    :caption: Staff Class 

    class Staff:
        def __init__(self,id,manager_name,absences,hire_date,authority_lvl,department,social_sec_no):
            self.manager_name = manager_name
            self.id = id
            self.absences = absences
            self.hire_date = hire_date
            self.authority_lvl = authority_lvl
            self.department = department
            self.social_sec_no = social_sec_no


Facility
+++++++++

.. code-block:: python
    :linenos:
    :caption: Facility Class

    class Facility:
        def __init__(self,id,campus_id,name,shortened_name,number_of_workers, size,expenses):
            self.id=id
            self.name = name
            self.shortened_name=shortened_name
            self.size=size
            self.number_of_workers=number_of_workers
            self.expenses = expenses
            self.campus_id = campus_id

Staff_facil
+++++++

.. code-block:: python
    :linenos:
    :caption: Staff_facil class`

    class Staff_facil:
        def __init__(self,title,from_date,to_date,salary,facility_id,staff_id,duty):
            self.title = title
            self.staff_id = staff_id
            self.to_date =to_date
            self.from_date = from_date
            self.salary = salary
            self.facility_id = facility_id
            self.duty = duty

View Models
-----------

View models handle GET/POST requests and render pages accordingly.

Models implemented by this user can be found in ``views.py`` file and shown below.


Staff
++++++
All operations for Staff Table and all CRUD operations for Staff-Facility Table are implemented in ``staff_add_page`` function as follows.


.. code-block:: python
    :linenos:
    :caption: Model for the Staff and Staff-Facility page

    def staff_add_page():
        db = current_app.config["db"]
        all_staff = db.get_all_staff()
        if request.method == "GET":
            return render_template("staff.html", staffs=all_staff, values=request.form)
    
        elif 'search_staff' in request.form:
            print("Searching staff.. id:", request.form.get("staff-id"))
            found_staff = db.get_staff(request.form.get("staff-id"))
            person_info = db.get_person(request.form.get("staff-id"))
            if found_staff is None:
                flash('No staff has been found.')
                return render_template("staff.html", staffs=all_staff,
                                       values=request.form)
            else:
                flash('Staff found!')
                return render_template("staff_search.html", staff=found_staff, staff_id=found_staff.id,
                                       values=request.form, person_info=person_info)
        elif 'delete_staff' in request.form:
    
            staff_id = request.form["staff_id"]
            print("Delete staff!", staff_id)
            db.delete_staff(int(staff_id))
            flash('Staff Deleted!')
            all_staff = db.get_all_staff()
            return render_template("staff.html", staffs=all_staff,
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
    
            new_staff = Staff(id=old_staff_id, manager_name=manager_name, absences=absences,
                              hire_date=hire_date,
                              social_sec_no=social_sec, department=department, authority_lvl=authority)
            db.update_staff(new_staff)
    
            flash('Staff Updated!')
            all_staff = db.get_all_staff()
            return render_template("staff.html",staffs=all_staff,
                                   values=request.form)
    
        elif 'more_info' in request.form:
            s_id = request.form["staff_id"]
            staff_facilities = db.get_facility_from_staff(s_id)
            the_staff = db.get_staff(s_id)
            facils = []
            for SF in staff_facilities:
                facility_ = db.get_facility(SF.facility_id)
                facils.append(facility_)
            if (len(facils) is None):
                flash('No facility information is given for this Staff')
            return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                   staff_facils=staff_facilities, values=request.form)
        elif 'add_staff_facil' in request.form:
            # Check validation
    
            s_id = request.form["staff_id"]
            staff_facilities = db.get_facility_from_staff(s_id)
            the_staff = db.get_staff(s_id)
            facils = []
            for SF in staff_facilities:
                facility_ = db.get_facility(SF.facility_id)
                facils.append(facility_)
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
                print("\nLength of staff_facilities array:", len(staff_facilities))
                facils = []
            except Error as e:
                flash('Staff-Facility Connection Not added!')
                if isinstance(e, errors.UniqueViolation):
                    flash('This connection already exists')
                    return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                           staff_facils=staff_facilities, values=request.form,
                                           error="ID already exists")
                if isinstance(e, errors.ForeignKeyViolation):
                    flash('Could not find the given staff Id or Facility Id')
                    return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                           staff_facils=staff_facilities, values=request.form,
                                           error="No ID")
    
                else:
                    return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                           staff_facils=staff_facilities, values=request.form,
                                           error=type(e).__name__ + "-----" + str(e))
            for SF in staff_facilities:
                facility_ = db.get_facility(SF.facility_id)
                facils.append(facility_)
            return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                   staff_facils=staff_facilities, values=request.form)
        elif 'delete_SF' in request.form:
            staff_id = request.form["staff_id_delete"]
            facility_id = request.form["facility_id_delete"]
            db.delete_staff_facil(int(staff_id), facility_id)
            flash('Staff-Facility Connection Deleted!')
            the_staff = db.get_staff(staff_id)
            staff_facilities = db.get_facility_from_staff(staff_id)
            facils = []
            for SF in staff_facilities:
                facility_ = db.get_facility(SF.facility_id)
                facils.append(facility_)
            return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                   staff_facils=staff_facilities, values=request.form)
    
        elif 'edit_SF' in request.form:
            staff_id = request.form["staff_id_edit"]
            staff = db.get_staff(staff_id)
            facility_id = request.form["facility_id_edit"]
            staff_facility = db.get_facility_from_staff(staff_id)
            facil = db.get_facility(facility_id)
            return render_template("edit_staff_facil.html", the_staff=staff, facility=facil,
                                   SF=staff_facility[0], values=request.form)
        elif 'edit_staff_facil' in request.form:
            title = request.form.get("title")
            from_date = request.form.get("from_date")
            to_date = request.form.get("to_date")
            salary = request.form.get("salary")
            facility_id = request.form.get("facility_id")
            staff_id = request.form.get("staff_id")
            duty = request.form.get("duty")
            new_SF = Staff_facil(title=title, from_date=from_date, to_date=to_date, salary=salary,
                                 facility_id=facility_id, staff_id=staff_id, duty=duty)
            db.update_SF(new_SF)
    
            flash('Staff-Facility Connection Updated!')
            all_SF = db.get_facility_from_staff(staff_id)
            the_staff = db.get_staff(staff_id)
            facils = []
            for SF in staff_facilities:
                facility_ = db.get_facility(SF.facility_id)
                facils.append(facility_)
            return render_template("staff_facility.html", staff=the_staff, facilities=facils,
                                   staff_facils=all_SF, values=request.form)
    
    
        else:  # Staff addition
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
                new_staff = Staff(id=staff_id, manager_name=manager_name, absences=absences,
                                  hire_date=hire_date, social_sec_no=social_sec, department=department,
                                  authority_lvl=authority)
                try:
                    db.add_staff(new_staff)
                    flash('Staff successfully added!')
                except Error as e:
                    flash('Staff NOT added!')
                    if isinstance(e, errors.UniqueViolation):
                        flash('A staff with this ID already exists')
                        return render_template("staff.html", form=request.form, staffs=all_staff,
                                               values=request.form,
                                               error="A staff with this ID already exists")
                    if isinstance(e, errors.ForeignKeyViolation):
                        flash('No people exists with this TR ID')
                        return render_template("staff.html", form=request.form, staffs=all_staff,
                                               values=request.form,
                                               error="No people exists with this TR ID")
    
                    else:
                        return render_template("staff.html", form=request.form, staffs=all_staff,
                                               values=request.form,
                                               error=type(e).__name__ + "-----" + str(e))
                return redirect(url_for("staff_add_page", staffs=all_staff, values=request.form))

.. code-block:: python
    :linenos:
    :caption: Form validation function for the Staff form

    def validation_staff(form):
        form.data = {}
        form.errors = {}
        db = current_app.config["db"]
    
        form_id = form.get("id")
        if db.get_staff(form_id):
            form.errors["id"] = "This staff is already registered with the given id."
            flash('This staff is already registered with the given id')
        elif form.get("id") == 0 or form.get("id") == None:
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
    

Staff-Facility
+++++++++
.. code-block:: python
    :linenos:
    :caption: Model for the Staff-Facility page

    def staff_facil_page():
        db = current_app.config["db"]
        all_staff = db.get_all_staff()
        if request.method == "GET":
            return render_template("staff.html", staffs=all_staff, values=request.form)

.. code-block:: python
    :linenos:
    :caption: Function for searching Campuses by Campus ID 

    def find_campus(campus_id):
        db = current_app.config["db"]
        campuses = db.get_campuses()
        for id, campus in campuses:
            if campus_id == id:
                return True
        return None


Facility
+++++++++

.. code-block:: python
    :linenos:
    :caption: Model for the Facility page

    def facility_page():
        db = current_app.config["db"]
        all_facilities = db.get_all_facility()
    
        if request.method == "GET":
            return render_template("facility.html", values=request.form, facilities=all_facilities)
    
    
        elif 'facility_search' in request.form:
            facil = db.get_facility(request.form.get("facility_id"))
            if facil is None:
                flash('No facility has been found.')
                return render_template("facility.html", facilities=all_facilities,
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
            campus = db.get_campus(campus_id)
            c_name = campus.name
            facilities = db.get_facility_from_campus(campus_id)
            if len(facilities) == 0:
                flash('There is no facility in this Campus.')
                return render_template("facility.html", facilities=all_facilities,
                                       values=request.form)
            return render_template("facility_search.html", facilities=facilities, campus_name=c_name,
                                   by_campus=1, values=request.form)
    
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
            return render_template("staff.html", staffs=all_staff,
                                   values=request.form)
    
        else:
            valid = validation_facility(request.form)
            if not valid:
                # flash('Input NOT Valid!')
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
                        return render_template("facility.html", form=request.form,
                                               facilities=all_facilities, values=request.form,
                                               error="A Facility with this ID already exists")
                    if isinstance(e, errors.ForeignKeyViolation):
                        flash('No campus exists with this ID')
                        return render_template("facility.html", form=request.form,
                                               facilities=all_facilities, values=request.form,
                                               error="No campus exists with this ID")
    
                    else:
                        return render_template("facility.html", form=request.form,
                                               facilities=all_facilities, values=request.form,
                                               error=type(e).__name__ + "-----" + str(e))
                return redirect(
                    url_for("facility_page", facilities=all_facilities, values=request.form))

.. code-block:: python
    :linenos:
    :caption: Form validation function for the Facility form

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




Database Queries
----------------

Database queries are handled via ``database.py`` file by constructing a Database class and using ``psycopg2`` library as the PostgreSQL driver.

Below are the related class methods implemented by this user:

Staff
++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Staff table

    def add_staff(self,staff):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "insert into staff (id, manager_name, absences, hire_date, authority_lvl,department, social_sec_no) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (staff.id, staff.manager_name, staff.absences, staff.hire_date, staff.authority_lvl, staff.department,
                                   staff.social_sec_no))
            connection.commit

    def get_staff(self,staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff where (id = %s)"
            cursor.execute(query, (staff_id,))
            if (cursor.rowcount == 0):
                return None
        found_staff = Staff(*cursor.fetchone()[:])
        return found_staff

    def get_all_staff(self):
        all_staff = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff order by id asc"
            cursor.execute(query)
            for row in cursor:
                staf = Staff(*row[:])
                all_staff.append(staf)
        return all_staff
    def delete_staff(self,staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from staff where (id = %s)"
            cursor.execute(query, (staff_id,))
            connection.commit

    def update_staff(self,staff):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update staff set  manager_name = %s, absences = %s, hire_date = %s, authority_lvl = %s,department = %s, social_sec_no = %s where (id = %s)"
            cursor.execute(query, ( staff.manager_name, staff.absences, staff.hire_date, staff.authority_lvl, staff.department,
                                   staff.social_sec_no, staff.id))
            connection.commit

Facility
+++++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Facility table

    def get_facility(self,facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility where (id = %s)"
            cursor.execute(query, (facility_id,))
            if (cursor.rowcount == 0):
                return None
        found_facility = Facility(*cursor.fetchone()[:])
        return found_facility
    def get_all_facility(self):
        all_facility = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility order by id asc"
            cursor.execute(query)
            for row in cursor:
                facil = Facility(*row[:])
                all_facility.append(facil)
        return all_facility
    def delete_facility(self,facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from facility where (id = %s)"
            cursor.execute(query, (facility_id,))
            connection.commit
    def add_facility(self,facility):
        with dbapi2.connect(self.dbfile) as connection:
            print("TRYİNG TO ADD:")
            print("----------")
            cursor = connection.cursor()
            query = "insert into facility (id, campus_id, name, shortened_name,number_of_workers,size,expenses) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (facility.id, facility.campus_id, facility.name, facility.shortened_name,
                                   facility.number_of_workers, facility.size, facility.expenses))
            connection.commit
    def get_facility_from_campus(self, campus_id):
        facilities = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from facility where (campus_id = %s) order by id asc"
            cursor.execute(query, (campus_id,))
            for row in cursor:
                facility = Facility(*row[:])
                facilities.append(facility)
        return facilities
    def update_facility(self,facility):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update facility set  id = %s, campus_id = %s, name = %s, shortened_name = %s,number_of_workers = %s, size = %s, expenses = %s where (id = %s)"
            cursor.execute(query, (facility.id, facility.campus_id, facility.name, facility.shortened_name, facility.number_of_workers,
                                   facility.size,
                                   facility.expenses))
            connection.commit

Staff-Facility
+++++++

.. code-block:: python
    :linenos:
    :caption: CRUD Operations for the Staff-Facility table

    def get_facility_from_staff(self, staff_id):
        staff_facilities = []
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff_facil where (staff_id = %s) order by staff_id asc"
            cursor.execute(query, (staff_id,))
            for row in cursor:
                SF = Staff_facil(*row[:])
                staff_facilities.append(SF)
        return staff_facilities
    def get_a_facility_from_staff(self, staff_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "select * from staff_facil where (staff_id = %s) order by staff_id asc"
            cursor.execute(query, (staff_id,))
            connection.commit

    def add_staff_facility(self,staff_facil):
        with dbapi2.connect(self.dbfile) as connection:
            print("TRYİNG TO ADD:")
            print("----------")
            cursor = connection.cursor()
            query = "insert into staff_facil (title,from_date,to_date,salary,facility_id,staff_id,duty) values (%s, %s, %s, %s, %s, %s,%s)"
            cursor.execute(query, (staff_facil.title, staff_facil.from_date, staff_facil.to_date, staff_facil.salary,
                                   staff_facil.facility_id, staff_facil.staff_id, staff_facil.duty))
            connection.commit
    def delete_staff_facil(self,staff_id, facility_id):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "delete from staff_facil where (staff_id= %s and facility_id = %s)"
            cursor.execute(query, (staff_id,facility_id))
            connection.commit

    def update_SF(self,SF):
        with dbapi2.connect(self.dbfile) as connection:
            cursor = connection.cursor()
            query = "update staff_facil set  title = %s, from_date = %s, to_date= %s, salary = %s, duty = %s where (staff_id = %s and facility_id = %s)"
            cursor.execute(query, ( staff_facil.title, staff_facil.from_date, staff_facil.to_date, staff_facil.salary,
                                   staff_facil.duty, staff_facil.staff_id, staff_facil.facility_id))
            connection.commit


Templates
---------

Following templates are written by **this user**:
    - ``edit_staff_facil.html``
    - ``facility.html``
    - ``facility_search.html``
    - ``staff.html``
    - ``staff_facility.html``
    - ``staff_search.html``


Contribution is made to the files below **this user** and **other teammates**:
    - ``layout.html``



