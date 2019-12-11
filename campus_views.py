from flask import current_app, render_template, request, redirect, url_for, abort

from forms import add_campus_form, add_faculty_form
from werkzeug.utils import secure_filename
import os
import io
from campus import Campus, Faculty
from base64 import b64encode

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'svg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def validate_image(image):
    return True


def campus():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    campus = {}
    form = add_campus_form()

    if request.method == "POST":
        if('delete_campus_flag' in request.form):
            campus_id = request.form['delete_campus_flag']
            print('campus id ->', campus_id)
            db.delete_campus(campus_id)
        else:
            if(form.validate()):
                image = request.files['image']
                if(validate_image(image)):
                    filename = secure_filename(image.filename)
                    file_extension = filename.split(".")[-1]
                    filename = filename.split(".")[0]
                    print('File name ->', filename)
                    print('\n File extension ->', file_extension)
                    # image.save(os.path.join(
                    #    current_app.config['UPLOAD_FOLDER'], filename))
                    # binary_img = open(os.path.join(
                    #    current_app.config['UPLOAD_FOLDER'], filename), 'rb')
                    byte_img = request.files['image'].read()
                    bin_img = ' '.join(map(bin, bytearray(byte_img)))
                    #content = binary_img.read()
                    campus = Campus(0, form.name.data, form.address.data, form.city.data, form.size.data,
                                    form.foundation_date.data, form.phone_number.data, filename, file_extension, bin_img)
                # os.remove(os.path.join(
                #      current_app.config['UPLOAD_FOLDER'], filename))
                    db.add_campus(campus)

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


def campus_detailed(campus_id):
    db = current_app.config["db"]
    campus = db.get_campus(campus_id)
    # if(campus is None):
    #     add_campus_form = add_campus_form()
    # else:
    #     add_campus_form = add_campus_form(
    #         {'name': campus.name, 'address': campus.address})
    # add_facultyForm = add_faculty_form()
    edit_campus_form = add_campus_form()
    add_faculty = add_faculty_form()
    if request.method == "POST" and 'change_picture' in request.form:
        file = request.files['image']
        file.save(secure_filename(file.filename))
        url = file.filename

        # if user does not select file, browser also
        # submit an empty part without filename
        if url == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(url):
            filename = secure_filename(url)
            file.save(os.path.join(
                current_app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('campus_detailed'))
    elif request.method == "POST" and 'add_faculty_form' in request.form:
        if(add_faculty_form.validate()):
            faculty = Faculty(request.form['add_faculty_form'], form.name.data, form.shortened_name.data,
                              form.address.data, form.foundation_date.data, form.phone_number.data)
            db.add_faculty(faculty)
    elif request.method == "POST" and 'edit_campus_form' in request.form:
        campus_id = campus.id
        updated_campus = Campus(campus_id, edit_campus_form.name.data, edit_campus_form.city.data, edit_campus_form.size.data,
                                edit_campus_form.foundation_date.data, edit_campus_form.phone_number.data, campus.image_name, campus.image_extension, campus.image_data)
        db.updated_campus(updated_campus)
    image = campus.img_data
    # print(bytearray(image))
    image = bytes(image)
    # print(image)
    image = b64encode(image)
    #print('zaa', image)
    context = {
        # 'add_faculty_form': add_facultyForm,
        'Campus': campus,
        'edit_campus_form': edit_campus_form,
        'campus_image': image,
        'add_faculty_form': add_faculty,
    }
    return render_template('/campuses/campus_detailed.html', context=context)


def findNumberOfCampus():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    return len(campuses)


def faculty_detailed():
    context = {}
    return render_template('/campuses/faculty_detailed.html', context=context)


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
