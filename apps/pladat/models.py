from django.db import models
from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from django.contrib.auth.models import User


class PladatUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True,
                                related_name='pladatuser')
    first_name = models.CharField(max_length=128, help_text='First name')
    last_name = models.CharField(max_length=128, help_text="Last name")
    phone_number = PhoneNumberField(
        help_text="Phone number")  # https://pypi.org/project/django-phonenumber-field/
    address = models.CharField(max_length=128, help_text="Addresss")
    city = models.CharField(max_length=128, help_text="City")
    state = models.CharField(max_length=128, null=True, help_text="State")
    country = CountryField(blank_label="Country")  # https://pypi.org/project/django-countries/

    class UserType(models.IntegerChoices):
        STUDENT = 0, 'Student account'
        COMPANY = 1, 'Company account'
        __empty__ = 'User type'

    user_type = models.IntegerField(choices=UserType.choices, help_text="User type")
