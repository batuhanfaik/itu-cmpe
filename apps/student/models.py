from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser

class Student(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete = models.CASCADE, primary_key = True)
    DEGREES = [
        ("bsc", "Bachelor of Science"),
        ("msc", "Master of Science"),
    ]
    degree = models.CharField(max_length=8, choices=DEGREES)
    MAJORS = [
        ("cmpe", "Computer Engineering"),
        ("math", "Mathematics"),
    ]
    major = models.CharField(max_length=8, choices=MAJORS)
    UNIVERSITIES = [
        ("itu", "Istanbul Technical University"),
        ("boun", "Bogazici University"),
        ("koc", "Koc University"),
    ]
    university = models.CharField(max_length=8, choices=UNIVERSITIES)
    number_of_previous_work_experience = models.IntegerField(default=0)
    years_worked = models.IntegerField(default=0)
    is_currently_employed = models.BooleanField(default=False)
    skills_text = models.TextField(max_length=512)
