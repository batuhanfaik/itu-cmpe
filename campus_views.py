from flask import current_app, render_template, request, redirect, url_for, abort
from flask_login import login_required, logout_user, login_user, current_user

from forms import add_campus_form, add_faculty_form, add_department_form
from werkzeug.utils import secure_filename
import os
import io
from campus import Campus, Faculty, Department
from base64 import b64encode
from psycopg2 import errors, Error

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'svg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@login_required
def validate_image(file_extension):
    if(file_extension in ALLOWED_EXTENSIONS):
        return True
    else:
        return False


@login_required
def campus():
    if(current_user.is_admin):
        db = current_app.config["db"]
        campuses = db.get_campuses()
        campus = {}
        form = add_campus_form()

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
            print('nenenene')

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
            print(form.errors)
            print(campus)
            db.add_campus(campus)
            return redirect(url_for('campus'))
        elif request.method == "POST" and "redirect_edit_page" in request.form:
            campus_form_id = request.form['redirect_edit_page']
            return redirect(url_for('campus_detailed', campus_id=campus_form_id))
            # img_name = secure_filename(image.filename)
            # print(bin_img)
            # agin = io.BytesIO(bin_img)

            # print(byte_img)

        context = {
            # 'form': form,
            'campuses': campuses,
            'form': form,
        }
        return render_template('/campuses/campus.html', context=context)


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
            if(validate_image(file_extension)):
                img_data = request.files['image'].read()
            updated_campus = Campus(campus_id, campus.name, campus.address, campus.city, campus.size,
                                    campus.foundation_date, campus.phone_number, file_extension, img_data)
            try:
                db.update_campus(updated_campus)
                return redirect(url_for('campus_detailed', campus_id=campus_id))
            except Error as e:
                error = type(e).__name__ + '----' + str(e)
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
            try:
                db.update_campus(updated_campus)
                return redirect(url_for('campus_detailed', campus_id=campus_id))
            except Error as e:
                error = type(e).__name__ + '----' + str(e)
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
            if(add_faculty.validate()):
                faculty = Faculty(0, request.form['add_faculty_form'], add_faculty.name.data, add_faculty.shortened_name.data,
                                  add_faculty.address.data, add_faculty.foundation_date.data, add_faculty.phone_number.data)
                try:
                    db.add_faculty(faculty)
                    return redirect(url_for('campus_detailed', campus_id=campus.id))
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
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
            if(edit_campus_form.validate()):
                try:
                    db.update_campus(updated_campus)
                    return redirect(url_for('campus_detailed', campus_id=campus.id))
                except Error as e:
                    error = type(e).__name__ + '----' + str(e)
                    pass
                add_faculty = add_faculty_form()
                edit_campus_form = add_campus_form()
                if('too long' in error):
                    error = "One of the input value is too long!"
            else:
                if('address' in edit_campus_form.errors):
                    error = 'Address field cannot be longer than 80 characters'
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
            print('ERROR RERERER', error)
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


@login_required
def findNumberOfCampus():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    return len(campuses)


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
                    add_error = type(e).__name__ + '----' + str(e)
                    pass
                context['add_error'] = add_error
            return render_template('/campuses/faculty_detailed.html', context=context, classrooms=classrooms)
        elif request.method == "POST" and 'edit_faculty_form' in request.form:
            print('heheryeherthethegr')

            if(edit_faculty_form.validate()):
                updated_faculty = Faculty(faculty_id, faculty.campus_id, edit_faculty_form.name.data, edit_faculty_form.shortened_name.data,
                                          edit_faculty_form.address.data, edit_faculty_form.foundation_date.data, edit_faculty_form.phone_number.data)
                try:
                    db.update_faculty(updated_faculty)
                    return redirect(url_for('faculty_detailed', faculty_id=faculty.id, classrooms=classrooms))
                except Error as e:
                    update_error = type(e).__name__ + '----' + str(e)
                    pass
                print('HEHEHEHE', update_error)
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
                cout << "error"
                return redirect(url_for('department_detailed', department_id=department.id))
        return render_template('/campuses/department_detailed.html', context=context)


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
