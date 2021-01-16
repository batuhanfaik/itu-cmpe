from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser


class Job(models.Model):
    title = models.CharField(max_length=128)
    description = models.TextField(max_length=512)
    requirements = models.TextField(max_length=512)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/
    # TODO: Enable this if it is true
    # recruiter = models.ManyToOneRel(PladatUser, limit_choices_to={'user_type':PladatUser.UserType.COMPANY})
    # Company info also required buy can be derived from recruiter