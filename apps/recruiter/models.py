from django.db import models
from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser, create_mock_pladatuser


class Recruiter(models.Model):
    pladatuser = models.OneToOneField(PladatUser, on_delete=models.CASCADE, primary_key=True, related_name='recruiter')
    company_name = models.CharField(max_length=8, null=True, help_text="Company Name")
    company_address = models.CharField(max_length=8, null=True, help_text="Company Address")
    company_phone_number = PhoneNumberField(
        help_text="Company Phone Number", null=True)  # https://pypi.org/project/django-phonenumber-field/

    @property
    def full_name(self):
        return "%s" % (self.pladatuser.full_name,)

    def __str__(self):
        return "Recruiter %s working for company %s" % (self.full_name, self.company_name)

def create_mock_recruiter(email = None, password = None):
    pladatuser = create_mock_pladatuser(email = email, password = password, user_type = PladatUser.UserType.RECRUITER)
    dct = {
        "pladatuser": pladatuser,
        "company_name": "ITU Rektorluk A.Ş",
        "company_address": "Maslak Ayazaga",
        "company_phone_number": "+447911123456"
    }
    recruiter = Recruiter.objects.create(**dct)
    return recruiter
