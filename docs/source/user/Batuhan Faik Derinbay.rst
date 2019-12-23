Parts Implemented by Batuhan Faik Derinbay
==========================================

Tables
------

Three tables, People, Student, Assistant tables, that are shown with examples below, have been implemented by this member.

People Table
    =========== ============ ======== ================= =================== ====== =============== ============= ============ ====== ========== ========== =========== =============== ========== =============== ==========
    tr_id       name         surname  phone_number      email               pass   person_category mother_fname  father_fname gender birth_city birth_date id_reg_city id_reg_district photo_name photo_extension photo_data
    ----------- ------------ -------- ----------------- ------------------- ------ --------------- ------------- ------------ ------ ---------- ---------- ----------- --------------- ---------- --------------- ----------
    11111111110 Batuhan Faik Derinbay +90 550 444 50 50 derinbay@itu.edu.tr SHA256 0               Mrs. Derinbay Mr. Derinbay M      Istanbul   01-01-2000 Istanbul    Sariyer         bfderinbay png             Base64
    =========== ============ ======== ================= =================== ====== =============== ============= ============ ====== ========== ========== =========== =============== ========== =============== ==========

*Holds the person information of every member signed up on the site.*

Assistant Table
    =========== ========== ============ ============ ============================= ==================== ======== ============= ========== ================= ===============
    tr_id       faculty_id supervisor   assistant_id bachelors                     degree               grad_gpa research_area office_day office_hour_start office_hour_end
    ----------- ---------- ------------ ------------ ----------------------------- -------------------- -------- ------------- ---------- ----------------- ---------------
    22222222220 3          Şule Öğüdücü 500150001    Istanbul Technical University Computer Engineering 4.0      Big Data      Monday     10:30             12:30
    =========== ========== ============ ============ ============================= ==================== ======== ============= ========== ================= ===============

*Holds the assistant related information of every assistant on the site.*

Student Table
    =========== ========== ============= ========== ======== ===== === ============= =====
    tr_id       faculty_id department_id student_id semester grade gpa credits_taken minor
    ----------- ---------- ------------- ---------- -------- ----- --- ------------- -----
    11111111110 1          2             150180705  5        2     4.0 69.5          False
    =========== ========== ============= ========== ======== ===== === ============= =====

*Holds the student related information of every student on the site.*

Design of the Website
---------------------

This user has implemented the design of the user interface, added, setup various frameworks and libraries including but not limited to Bootstrap, MDBootstrap, and JQuery.

Landing Page
++++++++++++

 .. figure:: ../../images/derinbay/index.png
    :alt: Index Page
    :align: center

    Landing page of ITU DataBees

Login Page
++++++++++

 .. figure:: ../../images/derinbay/login.png
    :alt: Login Page
    :align: center

    Login page of ITU DataBees

Pages of Tables
---------------

Related pages of the tables were also implemented.

In order to do any changes on the following tables the user must be logged in with an admin account. Student accounts don't any have permissions but to view the contents of the tables.

After logging in, admins can also re-initialize DataBees using the designated button on the landing page.

Every table has their dedicated pages to add, update and delete content.

Structure follows a relatively simple design with each table having one page to view all of its data as and add new data well as another page to view all of the attributes of a tuple and edit/delete the tuple.

People Page
+++++++++++

On this page, **admins** can add new users to the DataBees and every logged in user can see summary of people's attributes.

- TR IDs must be unique for each person.
- Emails must be unique for each person.
- Mother, father name, birth city and photo are optional fields.

 .. figure:: ../../images/derinbay/people.png
    :alt: People Page
    :align: center

    People page

Person Page
+++++++++++

On this page, **admins** can add edit existing users in the DataBees or delete them. Every logged in user can see all the attributes of a person.

- Update
    - TR IDs must be unique for each person.
    - Emails must be unique for each person.
    - Mother, father name, birth city and photo are optional fields.
- Delete
    - Since the People table holds data of users and is not weak every table that refers to is will lose its content upon deletion.
    - Used for removing users and people from DataBees.


 .. figure:: ../../images/derinbay/person.png
    :alt: Person Page
    :align: center

    Person page

Assistants Page
+++++++++++++++

On this page, **admins** can add new assistants to the DataBees and every logged in user can see summary of assistant's attributes.

- TR IDs must refer to existent people on People table.
- Faculty IDs must reference existent faculties on Faculty table.
- Supervisor must reference existent instructors' TR ID's on Instructor table.

 .. figure:: ../../images/derinbay/assistants.png
    :alt: Assistants Page
    :align: center

    Assistants page

Assistant Page
++++++++++++++

On this page, **admins** can add edit existing assistants in the DataBees or delete them. Every logged in user can see all the attributes of an assistant.

- Update
    - TR IDs must refer to existent people on People table.
    - Faculty IDs must reference existent faculties on Faculty table.
    - Supervisor must reference existent instructors' TR ID's on Instructor table.
- Delete
    - Assistant table is a weak table hence deletion of its tuples will not affect other tables.
    - Used for removing assistants from DataBees.


 .. figure:: ../../images/derinbay/assistant.png
    :alt: Assistant Page
    :align: center

    Assistant page

Students Page
+++++++++++++

On this page, **admins** can add new students to the DataBees and every logged in user can see summary of student's attributes.

- TR IDs must refer to existent people on People table.
- Faculty IDs must reference existent faculties on Faculty table.
- Department IDs must reference existent departments on Department table.

 .. figure:: ../../images/derinbay/students.png
    :alt: Students Page
    :align: center

    Students page

Student Page
++++++++++++

On this page, **admins** can add edit existing students in the DataBees or delete them. Every logged in user can see all the attributes of a student.

- Update
    - TR IDs must refer to existent people on People table.
    - Faculty IDs must reference existent faculties on Faculty table.
    - Department IDs must reference existent departments on Department table.
- Delete
    - Student table is a weak table hence deletion of its tuples will not affect other tables.
    - Used for removing students from DataBees.


 .. figure:: ../../images/derinbay/student.png
    :alt: Student Page
    :align: center

    Student page