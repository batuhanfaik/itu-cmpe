from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField


class Job(models.Model):
    title = models.CharField(max_length=128)
    description = models.TextField(max_length=512)
    requirements = models.TextField(max_length=512)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/