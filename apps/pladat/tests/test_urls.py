from django.test import SimpleTestCase
from django.urls import reverse, resolve
from apps.pladat.views import *


class TestUrls(SimpleTestCase):
    def setUp(self):
        pass

    def test_main_page_view(self):
        url = reverse("main_page_view")
        self.assertEqual(resolve(url).func, main_page_view)

    def test_registration_view(self):
        url = reverse("registration_view")
        self.assertEqual(resolve(url).func, registration_view)

    def test_login_page_view(self):
        url = reverse("login_page_view")
        self.assertEqual(resolve(url).func, login_page_view)

    def test_logout_page_view(self):
        url = reverse("logout_page_view")
        self.assertEqual(resolve(url).func, logout_page_view)

    def test_recruiter_profile_view(self):
        url = reverse("recruiter_profile_view")
        self.assertEqual(resolve(url).func, recruiter_profile_view)

    def recruiter_profile_update_view(self):
        url = reverse("recruiter_profile_update_view")
        self.assertEqual(resolve(url).func, recruiter_profile_update_view)