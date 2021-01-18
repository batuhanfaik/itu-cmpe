from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser, create_mock_pladatuser

class Skill(models.Model):
    name = models.CharField(max_length = 8, null = False, blank = False)
    def __str__(self):
        return self.name

class Student(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete = models.CASCADE, primary_key = True, related_name='student')
    skills = models.ManyToManyField(Skill, help_text="Skills")
    DEGREES = [
        ("bsc", "Bachelor of Science"),
        ("msc", "Master of Science"),
    ]
    degree = models.CharField(max_length=8, choices=DEGREES, null = True, help_text="Degree")
    MAJORS = [
        ("cmpe", "Computer Engineering"),
        ("math", "Mathematics"),
    ]
    major = models.CharField(max_length=8, choices=MAJORS, null = True, help_text="Major")
    UNIVERSITIES = [
        ("itu", "Istanbul Technical University"),
        ("boun", "Bogazici University"),
        ("koc", "Koc University"),
    ]
    university = models.CharField(max_length=8, choices=UNIVERSITIES, null = True, help_text="University")
    years_worked = models.PositiveIntegerField(default=0, help_text="Years Worked")

def create_mock_student(email = None, password = None):
    pladatuser = create_mock_pladatuser(email = email, password = password, user_type = PladatUser.UserType.STUDENT)
    dct = {
        "pladatuser": pladatuser,
        "degree": "bsc",
        "major": "cmpe",
        "university": "itu",
        "years_worked": 0
    }
    student = Student.objects.create(**dct)
    return student
