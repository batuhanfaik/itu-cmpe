from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser

class Skill(models.Model):
    name = models.CharField(max_length = 8, null = False, blank = False)
    def __str__(self):
        return self.name

class Student(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete = models.CASCADE, primary_key = True, related_name='student')
    skills = models.ManyToManyField(Skill)
    DEGREES = [
        ("bsc", "Bachelor of Science"),
        ("msc", "Master of Science"),
    ]
    degree = models.CharField(max_length=8, choices=DEGREES, null = True)
    MAJORS = [
        ("cmpe", "Computer Engineering"),
        ("math", "Mathematics"),
    ]
    major = models.CharField(max_length=8, choices=MAJORS, null = True)
    UNIVERSITIES = [
        ("itu", "Istanbul Technical University"),
        ("boun", "Bogazici University"),
        ("koc", "Koc University"),
    ]
    university = models.CharField(max_length=8, choices=UNIVERSITIES, null = True)
    number_of_previous_work_experience = models.IntegerField(default=0)
    years_worked = models.IntegerField(default=0)
    is_currently_employed = models.BooleanField(default=False)

