from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, BooleanField, RadioField
from wtforms.fields import DateField, DecimalField, FileField
from wtforms.validators import DataRequired, Length
from campus import Campus
from werkzeug.utils import secure_filename
# 'Adıyaman', 'Afyon', 'Ağrı', 'Amasya', 'Ankara', 'Antalya', 'Artvin',
#                        'Aydın', 'Balıkesir', 'Bilecik', 'Bingöl', 'Bitlis', 'Bolu', 'Burdur', 'Bursa', 'Çanakkale',
#                        'Çankırı', 'Çorum', 'Denizli', 'Diyarbakır', 'Edirne', 'Elazığ', 'Erzincan', 'Erzurum', 'Eskişehir',
#                        'Gaziantep', 'Giresun', 'Gümüşhane', 'Hakkari', 'Hatay', 'Isparta', 'Mersin', 'İstanbul', 'İzmir',
#                        'Kars', 'Kastamonu', 'Kayseri', 'Kırklareli', 'Kırşehir', 'Kocaeli', 'Konya', 'Kütahya', 'Malatya',
#                        'Manisa', 'Kahramanmaraş', 'Mardin', 'Muğla', 'Muş', 'Nevşehir', 'Niğde', 'Ordu', 'Rize', 'Sakarya',
#                        'Samsun', 'Siirt', 'Sinop', 'Sivas', 'Tekirdağ', 'Tokat', 'Trabzon', 'Tunceli', 'Şanlıurfa', 'Uşak',
#                        'Van', 'Yozgat', 'Zonguldak', 'Aksaray', 'Bayburt', 'Karaman', 'Kırıkkale', 'Batman', 'Şırnak',
#                        'Bartın', 'Ardahan', 'Iğdır', 'Yalova', 'Karabük', 'Kilis', 'Osmaniye', 'Düzce'

city_selections = [('Adana', 'Adana')]


class add_campus_form(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    address = TextAreaField('address', validators=[Length(min=4, max=50)])
    city = SelectField(choices=city_selections)
    size = DecimalField('size')
    foundation_date = DateField('foundation_date', validators=[
                                DataRequired()])
    phone_number = StringField('phone_number', validators=[
                               Length(10)])
    add_image_checkbox = BooleanField('add_image_checkbox')
    image = FileField('image')

    def save(form, image):
        print('HETYHEYHE', image)
        file_name = secure_filename(image.filename)
        # or ByteIO, whatever you like
        bin_file = image.read()
        print(bin_file)
        campus = Campus(0, form.name.data, form.address.data, form.city.data,
                        form.size.data, form.foundation_date.data, form.phone_number.data, file_name, image)
        validate_image(image)
        print('ADDEDDDDD->', name.data)
        print('Added to database')


# class upload_campus_image_form(FlaskForm):
#     image = FileField('image')

#     def save(form):
#         print('Added to database image')


class add_faculty_form(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    shortened_name = StringField('shortened_name', validators=[DataRequired()])
    address = TextAreaField('address', validators=[Length(min=4, max=50)])
    foundation_date = DateField('foundation_date')
    phone_number = StringField('phone_number')


class add_department_form(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    shortened_name = StringField('shortened_name')
    foundation_date = DateField('foundation_date')
    budget = DecimalField('budget')
 #   block_number = StringField('block_number')
    block_number = RadioField(
        "", choices=[('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')])

    phone_number = StringField('phone_number')


class login_form(FlaskForm):
    username = StringField("username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
