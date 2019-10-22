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
            campusname = request.form["campusname"]
            campuslocation = request.form["campuslocation"] 
            db.add_campus(Campus(campusname,campuslocation,[]))  
        elif 'edit_campus' in request.form:
            campusid = request.form["campusid"]
            campusname = request.form["campusname"]
            campuslocation = request.form["campuslocation"] 
            editedcampus = Campus(campusname,campuslocation,[])
            editedcampus.__setcampusid__(int(campusid))   
            db.edit_campus(editedcampus)      
        elif 'delete_campus' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()
            campusid = request.form["campusid"]
            db.delete_campus(int(campusid))
        elif 'add_faculty' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()    
            campusid = request.form["campusid"]
            facultyName = request.form["facultyName"]
            facultyShortenedName = request.form["facultyShortenedName"]
            db.add_faculty(int(campusid),Faculty(facultyName,facultyShortenedName))
        elif 'delete_faculty' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()
            campusid = request.form["campusid"]
            facultyid = request.form["facultyid"]
            db.delete_faculty(int(campusid),int(facultyid))
        elif 'edit_faculty' in request.form:
            campusid = request.form["campusid"]
            facultyid = request.form["facultyid"]
            facultyName = request.form["facultyName"]
            facultyShortenedName = request.form["facultyShortenedName"] 
            editedfaculty = Faculty(facultyName,facultyShortenedName)
            editedfaculty.__setfacultyid__(int(facultyid))   
            db.edit_faculty(editedfaculty)   

        return redirect(url_for("manage_campusses", campusses=campusses))
