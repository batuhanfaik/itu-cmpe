from flask import current_app, render_template,request,redirect,url_for
from campus import Campus
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
            db.add_campus(Campus(campusname,campuslocation))  
        elif 'edit_campus' in request.form:
            campusid = request.form["campusid"]
            campusname = request.form["campusname"]
            campuslocation = request.form["campuslocation"] 
            editedcampus = Campus(campusname,campuslocation)
            editedcampus.__setcampusid__(int(campusid))   
            db.edit_campus(editedcampus)      
        elif 'delete_campus' in request.form:
            db = current_app.config["db"]
            campusses = db.get_campusses()
            campusid = request.form["campusid"]
            db.delete_campus(int(campusid))
        return redirect(url_for("manage_campusses", campusses=campusses))
