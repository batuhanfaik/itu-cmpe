from django.test import TestCase

from apps.student.models import *
from apps.pladat.models import *

class TestModels(TestCase):
    def setUp(self):
        self.skill = Skill.objects.create(name="harika muhendislik")
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
        self.student = Student.objects.create(
            pladatuser=self.pladat_user,
            degree='bsc',
            major='cmpe',
            university='itu',
            years_worked=5,
        )

    def test_student_created(self):
        self.assertEquals(self.student, Student.objects.get(pladatuser=self.pladat_user))

    def test_skill_created(self):
        self.assertEquals(self.skill, Skill.objects.get(name='harika muhendislik'))