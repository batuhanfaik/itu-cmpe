from django.db import models
from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from django.contrib.auth.models import User

class PladatUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    first_name = models.CharField(max_length=128)
    last_name = models.CharField(max_length=128)
    phone_number = PhoneNumberField()  # https://pypi.org/project/django-phonenumber-field/
    address = models.CharField(max_length=128)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/
    
    STUDENT = 0
    COMPANY = 1
    USER_TYPE = [
        (STUDENT, 'Student account'),
        (COMPANY, 'Company account'),
    ]
    user_type = models.IntegerField(choices=USER_TYPE)

