from django.db import models
from django_countries.fields import CountryField


class Student(models.Model):
    first_name = models.CharField(max_length=128)
    last_name = models.CharField(max_length=128)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()    # https://pypi.org/project/django-countries/
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


class Job(models.Model):
    title = models.CharField(max_length=128)
    description = models.TextField(max_length=512)
    requirements = models.TextField(max_length=512)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/
