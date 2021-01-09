from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser

class Recruiter(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete = models.CASCADE, primary_key = True)