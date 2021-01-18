from django.contrib.auth.models import User
from django.db import models
from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField


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
    country = CountryField(blank_label="Country", help_text="Country")  # https://pypi.org/project/django-countries/

    class UserType(models.IntegerChoices):
        STUDENT = 0, 'Student account'
        RECRUITER = 1, 'Recruiter account'
        __empty__ = 'User type'

    user_type = models.IntegerField(choices=UserType.choices, help_text="User type")

    def is_student(self):
        return self.user_type == self.UserType.STUDENT

    @property
    def full_name(self):
        return "%s %s" % (self.first_name, self.last_name)


def random_str(len):
    # Returns a random string of length len
    import random
    import string
    return ''.join(random.choice(string.ascii_letters) for _ in range(len))


def create_mock_pladatuser(email = None, password = None, user_type = None, user = None):
    dct = {
        "first_name": "Isim2",
        "last_name": "Soyisim2",
        "phone_number": "+447911123456",
        "address": "Maslak Ayazaga",
        "city": "Istanbul",
        "state": "Pennsylvania",
        "country": "TR",
    }

    if not email:
        email = "%s@hotmail.com" % (random_str(7), )
    
    if not password:
        password = random_str(5)

    if user_type:
        dct['user_type'] = user_type
    else:
        dct['user_type'] = 0

    if user:
        dct['user'] = user
    else:
        user_dct = {
            'username': email,
            'email': email,
            'password': password,
        }
        user = User.objects.create_user(**user_dct)
        user.save()
        dct['user'] = user
    
    pladatuser = PladatUser.objects.create(**dct)
    
    return pladatuser




