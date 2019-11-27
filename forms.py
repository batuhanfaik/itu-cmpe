from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.fields import DateField, DecimalField, FileField
from wtforms.validators import DataRequired


class add_campus_form(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    address = StringField('address')
    city = StringField('city')
    foundation_date = DateField('foundation_date')
    size = DecimalField('size')
    phone_number = StringField('phone_number')


class upload_campus_image_form(FlaskForm):
    image = FileField()


class login_form(FlaskForm):
    identity = StringField("identity", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
