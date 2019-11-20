from flask_wtf import FlaskForm, Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired
from wtforms.fields import DateField,DecimalField,FileField,TextAreaField
import wtforms.validators as validators

class add_campus_form(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    address = StringField('address')
    city = StringField('city')
    foundation_date = DateField('foundation_date')
    size = DecimalField('size')
    phone_number = StringField('phonu_number')

class upload_campus_image_form(FlaskForm):
    image = FileField()

class login_form(FlaskForm):
    email = StringField("email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
