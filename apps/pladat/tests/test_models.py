from django.test import TestCase
from apps.pladat.models import *


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
            user_type=0,
        )

    def test_user_created(self):
        self.assertEquals(self.test_user, User.objects.get(email="test@pladat.com"))

    def test_create_pladat_user(self):
        self.assertEquals(self.pladat_user, PladatUser.objects.get(user=self.test_user))

    def test_delete_pladat_user(self):
        self.pladat_user.delete()
        with self.assertRaises(PladatUser.DoesNotExist):
            _ = PladatUser.objects.get(user=self.test_user)
