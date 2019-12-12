from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, BooleanField, RadioField
from wtforms.fields import DateField, DecimalField, FileField
from wtforms.validators import DataRequired, Length
from campus import Campus
from werkzeug.utils import secure_filename

# 'Bartın', 'Ardahan', 'Iğdır', 'Yalova', 'Karabük', 'Kilis', 'Osmaniye', 'Düzce'

city_selections = [('Adana', 'Adana'),('Adıyaman', 'Adıyaman'),('Ağrı', 'Ağrı'),('Amasya', 'Amasya'),('Ankara', 'Ankara'),('Antalya', 'Antalya'),('Artvin', 'Artvin'),('Aydın', 'Aydın'),('Balıkesir', 'Balıkesir'),('Bilecik', 'Bilecik'),('Bingöl', 'Bingöl'),('Bitlis', 'Bitlis'),('Bolu', 'Bolu'),('Burdur', 'Burdur'),('Bursa', 'Bursa'),('Çanakkale', 'Çanakkale'),('Çankırı', 'Çankırı'),('Çorum', 'Çorum'),('Denizli', 'Denizli'),('Diyarbakır', 'Diyarbakır'),('Edirne', 'Edirne'),('Elazığ', 'Elazığ'),('Erzincan', 'Erzincan'),('Erzurum', 'Erzurum'),('Eskişehir', 'Eskişehir'),('Gaziantep', 'Gaziantep'),('Giresun', 'Giresun'),('Gümüşhane', 'Gümüşhane'),('Hakkari', 'Hakkari'),('Hatay', 'Hatay'),('Isparta', 'Isparta'),('Mersin', 'Mersin'),('İstanbul', 'İstanbul'),('İzmir', 'İzmir'),('Kars', 'Kars'),('Kastamonu', 'Kastamonu'),('Kayseri', 'Kayseri'),('Kırklareli', 'Kırklareli'),('Kırşehir', 'Kırşehir'),('Kocaeli', 'Kocaeli'),('Konya', 'Konya'),('Kütahya', 'Kütahya'),('Malatya', 'Malatya'),('Manisa', 'Manisa'),('Kahramanmaraş', 'Kahramanmaraş'),('Mardin', 'Mardin'),('Muğla', 'Muğla'),('Muş', 'Muş'),('Nevşehir', 'Nevşehir'),
('Niğde', 'Niğde'),('Ordu', 'Ordu'),('Rize', 'Rize'),('Sakarya', 'Sakarya'),('Samsun', 'Samsun'),('Siirt', 'Siirt'),('Sinop', 'Sinop'),('Sivas', 'Sivas'),('Tekirdağ', 'Tekirdağ'),('Tokat', 'Tokat'),
('Trabzon', 'Trabzon'),('Tunceli', 'Tunceli'),('Şanlıurfa', 'Şanlıurfa'),('Uşak', 'Uşak'),('Van', 'Van'),('Yozgat', 'Yozgat'),('Zonguldak', 'Zonguldak'),('Aksaray', 'Aksaray'),('Bayburt', 'Bayburt'),('Karaman', 'Karaman'),('Kırıkkale', 'Kırıkkale'),('Batman', 'Batman'),('Şırnak', 'Şırnak'),
('Bartın', 'Bartın'),
('Ardahan', 'Ardahan'),
('Iğdır', 'Iğdır'),
('Yalova', 'Yalova'),
('Karabük', 'Karabük'),
('Kilis', 'Kilis'),
('Osmaniye', 'Osmaniye'),
('Düzce', 'Düzce'),
('Abroad', 'Abroad')]


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
