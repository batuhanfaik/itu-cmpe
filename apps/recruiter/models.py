from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser


class Recruiter(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete=models.CASCADE, primary_key=True, related_name='recruiter')
    company_name = models.CharField(max_length=8, null=True, help_text="Company Name")
    company_address = models.CharField(max_length=8, null=True, help_text="Company Address")
    company_phone_number = PhoneNumberField(
        help_text="Company Phone Number", null=True)  # https://pypi.org/project/django-phonenumber-field/


