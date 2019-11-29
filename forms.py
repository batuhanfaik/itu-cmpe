from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.fields import DateField, DecimalField, FileField
from wtforms.validators import DataRequired


class add_campus_form(FlaskForm):
    nm = ''
    addr = ''
    cty = ''
    fd = ''
    sze = ''
    phnu = ''

    def __init__(self, *args, **kwargs):
        self.nm = args[0]['name']
        self.cty = args[0]['address']
        self.addr = args[0]['address']
        self.foundation_date = args[0]['foundation_date']
        self.size = args[0]['size']
        self.phone_number = args[0]['phone_number']
        super(add_campus_form, self).__init__(*args, **kwargs)

    name = StringField('name', validators=[DataRequired()])
    address = StringField('address')
    city = StringField('city')
    foundation_date = DateField('foundation_date')
    size = DecimalField('size')
    phone_number = StringField('phone_number')


class add_faculty_form(FlaskForm):
    nm = ''
    short_nm = ''
    addr = ''
    fd = ''
    phnu = ''

    def __init__(self, *args, **kwargs):
        self.nm = args[0]['name']
        self.short_nm = args[0]['shortened_name']
        self.addr = args[0]['address']
        self.foundation_date = args[0]['foundation_date']
        self.phone_number = args[0]['phone_number']
        super(add_campus_form, self).__init__(*args, **kwargs)

    name = StringField('name', validators=[DataRequired()])
    address = StringField('address')
    city = StringField('city')
    foundation_date = DateField('foundation_date')
    size = DecimalField('size')
    phone_number = StringField('phone_number')


class add_department_form(FlaskForm):
    fclty_id = ''
    nm = ''
    short_nm = ''
    block_n = ''
    bdgt = ''
    fd = ''
    phnu = ''

    def __init__(self, *args, **kwargs):
        self.fclty_id = args[0]['faculty_id']
        self.nm = args[0]['name']
        self.short_nm = args[0]['shortened_name']
        self.block_n = args[0]['block_number']
        self.bdgt = args[0]['budget']
        self.fd = args[0]['foundation_date']
        self.phnu = args[0]['phone_number']
        super(add_campus_form, self).__init__(*args, **kwargs)

    name = StringField('name', validators=[DataRequired()])
    address = StringField('address')
    city = StringField('city')
    foundation_date = DateField('foundation_date')
    size = DecimalField('size')
    phone_number = StringField('phone_number')


class upload_campus_image_form(FlaskForm):
    image = FileField()


class login_form(FlaskForm):
    username = StringField("username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
