from flask import current_app, render_template, request, redirect, url_for, abort

from forms import add_campus_form, add_faculty_form
from werkzeug.utils import secure_filename
import os
import io
from campus import Campus
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
    print('oc')
    return True


def campus():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    campuses = {}
    campus = {}
    form = add_campus_form()
    if request.method == "POST":
        if(form.validate()):
            image = request.files['image']
            if(validate_image(image)):
                filename = secure_filename(image.filename)
                image.save(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], filename))
                binary_img = open(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], filename), 'rb')
                content = binary_img.read()
                campus = Campus(form.name.data, form.address.data, form.city.data, form.size.data,
                                form.foundation_date.data, form.phone_number.data, filename, content)
                os.remove(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], filename))
                db.add_campus(campus)
            # byte_img = request.files['image'].read()
                #bin_img = ' '.join(map(bin, bytearray(byte_img)))
            # print(bin_img)

            #img_name = secure_filename(image.filename)
            # print(bin_img)
                #agin = io.BytesIO(bin_img)

                # print(byte_img)

    context = {
        # 'form': form,
        'campuses': campuses,
        'form': form,
    }
    return render_template('/campuses/campus.html', context=context)


def campus_detailed():
    # db = current_app.config["db"]
    # campus = db.get_campus(0)
    # if(campus is None):
    #     add_campus_form = add_campus_form()
    # else:
    #     add_campus_form = add_campus_form(
    #         {'name': campus.name, 'address': campus.address})
    # add_facultyForm = add_faculty_form()
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
            return redirect(url_for('campus_detailed'))

    if request.method == "POST" and request.form.validate():
        faculty = Faculty()
        return redirect(url_for('home'))
    context = {
        # 'add_faculty_form': add_facultyForm,
        'campus': campus,
    }
    return render_template('/campuses/campus_detailed.html', context=context)


def findNumberOfCampus():
    db = current_app.config["db"]
    campuses = db.get_campuses()
    return len(campuses)


def faculty_detailed():
    context = {}
    return render_template('/campuses/faculty_detailed.html', context=context)


def edit_campus(request):
    db = current_app.config["db"]
    add_campus = False
    campus_id = request.form.campus_id
    if request.method == "GET":
        campus = db.get_campus(request.args.get('campus_id'))

    context = {
        'form': form,
        'add': False,
        'campus': campus
    }
    return render_template('/campuses/campus.html', context=context)


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
