from django.test import TestCase, Client
from django.urls import reverse
from apps.pladat.models import *


class TestViews(TestCase):
    def setUp(self):
        self.client = Client()
        self.main_page_url = reverse("main_page_view")
        self.login_page_url = reverse("login_page_view")
        self.test_user = User.objects.create_user(
            username="test@pladat.com",
            email="test@pladat.com",
            password="password",
        )

    def test_main_page_view_GET(self):
        response = self.client.get(self.main_page_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "main_page.html")

    def test_login_page_view_GET(self):
        response = self.client.get(self.login_page_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "user_login.html")

    def test_login_page_view_POST_successful_user_login(self):
        login_data = {
            "email": "test@pladat.com",
            "password": "password",
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertEquals(response.status_code, 302)

    def test_login_page_view_POST_unsuccessful_user_login(self):
        login_data = {
            "email": "test@pladat.com",
            "password": "WRONGpassword",
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "user_login.html")
