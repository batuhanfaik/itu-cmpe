from django.test import TestCase

from apps.pladat.models import *
from apps.job.models import *


class TestModel(TestCase):
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
            user_type=1,
        )
        self.recruiter = Recruiter.objects.create(
            pladatuser=self.pladat_user,
            company_name="Test Company",
            company_address="Maslak Ayazaga",
            company_phone_number="+902124440911",
        )
        self.test_job = Job.objects.create(
            title="Test Job",
            description="This is a test job for students to apply.",
            city="Istanbul",
            state=2,
            country="TR",
            recruiter=self.recruiter,
        )

    def test_user_created(self):
        self.assertEquals(self.test_user, User.objects.get(email="test@pladat.com"))

    def test_create_pladat_user(self):
        self.assertEquals(self.pladat_user, PladatUser.objects.get(user=self.test_user))

    def test_create_recruiter(self):
        self.assertEquals(self.recruiter, Recruiter.objects.get(pladatuser=self.pladat_user))

    def test_create_job(self):
        self.assertEquals(self.test_job,
                          Job.objects.get(title="Test Job", recruiter=self.recruiter))

    def test_delete_job(self):
        self.test_job.delete()
        with self.assertRaises(Job.DoesNotExist):
            _ = Job.objects.get(title="Test Job", recruiter=self.recruiter)
