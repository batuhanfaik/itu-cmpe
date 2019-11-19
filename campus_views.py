from flask import current_app, render_template, request, redirect, url_for, abort

from forms import upload_campus_image_form, add_campus_form

def campus():
    db = current_app.config["db"]
    form = add_campus_form()
    add_campus = True
    if request.method == "POST" and form.validate():
        campus = Campus(form.name,form.address,form.city,form.foundation_date,form.size)
        return redirect(url_for('home'))
    context ={
        'form': form,
        'add' : True
    }
    return render_template('/campuses/campus.html',context = context)

def edit_campus(request):
    db = current_app.config["db"]
    add_campus = False
    campus_id = request.form.campus_id
    if request.method == "GET":
        campus = db.get_campus(request.args.get('campus_id'))

    context = {
        'form' : form,
        'add' : False,
        'campus' : campus
    }
    return render_template('/campuses/campus.html',context = context)


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
