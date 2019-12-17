from flask import current_app
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SelectField, BooleanField, RadioField
from wtforms.fields import DateField, DecimalField, FileField, IntegerField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Length, InputRequired
from wtforms import ValidationError
from wtforms_components import TimeField
from campus import Campus
from werkzeug.utils import secure_filename


class SelectCourseForm(FlaskForm):
    crn1 = IntegerField("CRN1", render_kw={'type': 'number', 'class': 'crn'}, validators=[InputRequired()])
    crn2 = IntegerField("CRN2", render_kw={'type': 'number', 'class': 'crn'}, default=0)
    crn3 = IntegerField("CRN3", render_kw={'type': 'number', 'class': 'crn'}, default=0)
    crn4 = IntegerField("CRN4", render_kw={'type': 'number', 'class': 'crn'}, default=0)
    crn5 = IntegerField("CRN5", render_kw={'type': 'number', 'class': 'crn'}, default=0)
    crn6 = IntegerField("CRN6", render_kw={'type': 'number', 'class': 'crn'}, default=0)


class CourseForm(FlaskForm):
    crn = StringField("CRN", validators=[InputRequired(), Length(min=5, max=5)])
    code = StringField("Code", validators=[InputRequired(), Length(min=3, max=3)])
    name = StringField("Course Name", validators=[InputRequired(), Length(max=100)])
    start_time = TimeField("Start Time", validators=[InputRequired()])
    end_time = TimeField("End Time", validators=[InputRequired()])
    def validate_end_time(self, field):
        if self.start_time.data >= field.data:
            raise ValidationError("End time can not be equal or lower than Start Time")
    day = SelectField("Day", choices=[('Monday', 'Monday'), ('Tuesday', 'Tuesday'),
                                      ('Wednesday', 'Wednesday'), ('Thursday', 'Thursday'),
                                      ('Friday', 'Friday')])
    capacity = IntegerField("Capacity", validators=[InputRequired()], render_kw={'type': 'number'})
    enrolled = HiddenField("Enrolled", default=0)
    credits = DecimalField("Credits", validators=[InputRequired()], places=1)
    language = StringField("Language", validators=[Length(max=2)])
    classroom_id = IntegerField("Classroom ID", validators=[InputRequired()], render_kw={'type': 'number'})
    instructor_id = IntegerField("Instructor ID", validators=[InputRequired()], render_kw={'type': 'number'})
    department_id = IntegerField("Department ID", validators=[InputRequired()], render_kw={'type': 'number'})
    info = TextAreaField("Course Information", validators=[InputRequired()])


class ClassroomForm(FlaskForm):
    faculty_id = None
    capacity = IntegerField(u"Capacity", validators=[InputRequired()], render_kw={'type': 'number'})
    door_number = StringField(u"Door Number", validators=[InputRequired(), Length(max=4)])
    # def validate_door_number(self, field):
    #     db = current_app.config['db']
    #     classroom = db.get_classroom_by_door_and_faculty(self.faculty_id, field.data)
    #     if classroom is not None:
    #         raise ValidationError("There exists is a classroom with this door number!")
    floor = StringField(u"Floor", validators=[Length(max=2)])
    board_count = IntegerField(u"Board Count", render_kw={'type': 'number'})
    has_projection = RadioField(u"Has Projection", choices=[('true', 'Yes'), ('false', 'No')], default='false')
    # has_projection = BooleanField(u"Has Projection")
    renewed = BooleanField(u"Renewed")
    air_conditioner = BooleanField(u"Has Air Conditioner")


class InstructorForm(FlaskForm):
    tr_id = IntegerField(u"TR ID", validators=[InputRequired()], render_kw={'type': 'number'}) # , Length(min=11, max=11, message="TR ID's length must be 11")
    department_id = IntegerField(u"Department ID", validators=[InputRequired()], render_kw={'type': 'number'})
    faculty_id = IntegerField(u"Faculty ID", validators=[InputRequired()], render_kw={'type': 'number'})
    specialization = StringField(u"Specialization")
    bachelors = StringField(u"Bachelor's Degree")
    masters = StringField(u"Master's Degree")
    doctorates = StringField(u"Doctorate's Degree")
    room_id = StringField(u"Room ID", validators=[InputRequired(), Length(max=4)])


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
    name = StringField('name', validators=[DataRequired(),Length(max=50)])
    address = TextAreaField('address', validators=[DataRequired(),Length(max=80)])
    city = SelectField(choices=city_selections)
    size = DecimalField('size')
    foundation_date = DateField('foundation_date', validators=[])
    phone_number = StringField('phone_number', validators=[])
    add_image_checkbox = BooleanField('add_image_checkbox')
    image = FileField('image',)

    def save(form, image):
        print('HETYHEYHE', image)
        file_name = secure_filename(image.filename)
        # or ByteIO, whatever you like
        bin_file = image.read()
        campus = Campus(0, form.name.data, form.address.data, form.city.data,
                        form.size.data, form.foundation_date.data, form.phone_number.data, file_name, image)
        validate_image(image)


# class upload_campus_image_form(FlaskForm):
#     image = FileField('image')

#     def save(form):
#         print('Added to database image')


class add_faculty_form(FlaskForm):
    name = StringField('name', validators=[DataRequired(),Length(max=100)])
    shortened_name = StringField('shortened_name', validators=[DataRequired()])
    address = TextAreaField('address', validators=[Length(max=80)])
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
