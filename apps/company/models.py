from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

class Recruiter(models.Model):
    first_name = models.CharField(max_length=128)
    last_name = models.CharField(max_length=128)
    email = models.EmailField(max_length=128)
    phone_number = PhoneNumberField()  # https://pypi.org/project/django-phonenumber-field/
    title = models.CharField(max_length=128)
    address = models.CharField(max_length=128)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/