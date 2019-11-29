from flask import current_app, render_template, request, redirect, url_for, abort

from forms import upload_campus_image_form, add_campus_form, add_faculty_form


def campus():
    #db = current_app.config["db"]
    #campuses = db.get_campuses()
    campuses = {}
    campus = {'g': 's'}
   # form = add_campus_form({'name': '', 'city': '', 'address': '',
 #                           'foundation_date': '', 'size': '', 'phone_number': ''})
    if request.method == "POST" and form.validate():
        campus = Campus(form.name, form.address, form.city,
                        form.foundation_date, form.size)
        return redirect(url_for('home'))
    context = {
        # 'form': form,
        'campuses': campuses,
        'Campus': campus,
    }
    return render_template('/campuses/campus.html', context=context)


def campus_detailed():
    #db = current_app.config["db"]
    #campus = db.get_campus(0)
    # if(campus is None):
    #     add_campus_form = add_campus_form()
    # else:
    #     add_campus_form = add_campus_form(
    #         {'name': campus.name, 'address': campus.address})
    #add_facultyForm = add_faculty_form()
    if request.method == "POST" and form.validate():
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
            imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('list_campus',
                                    filename=filename))
    return form
