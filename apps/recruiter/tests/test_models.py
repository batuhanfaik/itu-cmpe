from django.test import TestCase

from apps.recruiter.models import *
from apps.pladat.models import *

class TestModels(TestCase):
    def setUp(self):
        self.test_user = User.objects.create_user(
            username="test@pladat.com",
            email="test@pladat.com",
            password="password",
        )
        self.pladat_user = PladatUser.objects.create(
            user=self.test_user,
            first_name="Isim",
            last_name="Soyisim",
            phone_number="+447911123456",
            address="Maslak Ayazaga",
            city="Istanbul",
            state=2,
            country="TR",
            user_type=0,
        )
        self.recruiter = Recruiter.objects.create(
            pladatuser=self.pladat_user,
            company_name="Testing Company",
            company_address="Maslak",
            company_phone_number="+905555555555"
        )

    def test_recruiter_created(self):
        self.assertEquals(self.recruiter, Recruiter.objects.get(pladatuser=self.pladat_user))

    def test_full_name(self):
        self.assertEquals(self.recruiter.full_name, "Isim Soyisim")