from flask import current_app, render_template,request,redirect,url_for
from campus import Campus
from faculty import Faculty
from database import Database

def manage_campusses(): 
    db = current_app.config["db"]
    campusses = db.get_campusses()
    if request.method=="GET":
        return render_template("manage_campusses.html", campusses=campusses)
    else:
        if 'add_campus' in request.form:
            campus_name = request.form["campus_name"]
            campus_location = request.form["campus_location"] 
            db.add_campus(Campus(campus_name,campus_location,[]))  
        elif 'edit_campus' in request.form:
            campus_id = request.form["campus_id"]
            campus_name = request.form["campus_name"]
            campus_location = request.form["campus_location"] 
            edited_campus = Campus(campus_name,campus_location,campusses.get(campus_id).faculties)
            edited_campus.set_campus_id(int(campus_id))   
            db.edit_campus(edited_campus)      
        elif 'delete_campus' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()
            campus_id = request.form["campus_id"]
            db.delete_campus(int(campus_id))
        elif 'add_faculty' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()    
            campus_id = request.form["campus_id"]
            faculty_name = request.form["faculty_name"]
            faculty_shortened_name = request.form["faculty_shortened_name"]
            db.add_faculty(int(campus_id),Faculty(faculty_name,faculty_shortened_name))
        elif 'delete_faculty' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()
            campus_id = request.form["campus_id"]
            faculty_id = request.form["faculty_id"]
            db.delete_faculty(int(campus_id),int(faculty_id))
        elif 'edit_faculty' in request.form:
            campus_id = request.form["campus_id"]
            faculty_id = request.form["faculty_id"]
            faculty_name = request.form["faculty_name"]
            faculty_shortened_name = request.form["faculty_shortened_name"] 
            edited_faculty = Faculty(faculty_name,faculty_shortened_name)
            edited_faculty.set_faculty_id(int(faculty_id))   
            db.edit_faculty(edited_faculty)   

        return redirect(url_for("manage_campusses", campusses=campusses))
