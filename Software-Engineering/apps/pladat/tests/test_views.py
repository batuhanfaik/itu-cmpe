from django.test import Client, TestCase
from django.urls import reverse

from apps.pladat.views import *


class TestViews(TestCase):
    def setUp(self):
        self.client = Client()
        self.main_page_url = reverse("main_page_view")
        self.login_page_url = reverse("login_page_view")
        self.logout_page_url = reverse("logout_page_view")
        self.register_page_url = reverse("registration_view")
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
        self.assertEqual(response.status_code, 302)

    def test_login_page_view_forbidden_method(self):
        response = self.client.put(self.login_page_url, data={})
        self.assertEqual(response.status_code, 403)

    def test_logout_page_view_POST_successful_login_logout_redirect(self):
        login_data = {
            "email": "test@pladat.com",
            "password": "password",
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertEqual(response.status_code, 302)
        response = self.client.post(self.logout_page_url, data={})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

    def test_logout_page_view_POST_unsuccessful_no_login_logout(self):
        response = self.client.post(self.logout_page_url, data={})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

    def test_login_page_view_POST_unsuccessful_no_user(self):
        login_data = {
            "email": "this@email.com",
            "password": "pass"
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertTemplateUsed(response, "user_login.html")

    def test_login_page_view_POST_unsuccessful_user_login(self):
        login_data = {
            "email": "thisemaildoesnotexist",
            "password": "pass"
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertTemplateUsed(response, "user_login.html")

    def test_register_view_GET_no_user_logged_in(self):
        response = self.client.get(self.register_page_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "user_register.html")

    def test_register_view_GET_user_already_logged_in(self):
        login_data = {
            "email": "test@pladat.com",
            "password": "password",
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertEqual(response.status_code, 302)
        response = self.client.get(self.register_page_url)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, '/')

    def test_register_view_POST_successful_registration(self):
        register_data = {
            'email': 'test@test.com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        }
        response = self.client.post(self.register_page_url, register_data)
        self.assertTemplateUsed(response, 'user_register.html')
        self.assertEqual(response.status_code, 200)

    def test_register_view_POST_unsuccessful_registration_used_email(self):
        register_data = {
            'email': 'test@pladat.com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        }
        response = self.client.post(self.register_page_url, register_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'user_register.html')

    def test_register_view_POST_unsuccessful_registration_invalid_email(self):
        register_data = {
            'email': 'test@com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        }
        response = self.client.post(self.register_page_url, register_data)
        self.assertFalse(User.objects.filter(email='test@com').exists())
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'user_register.html')

    def test_register_view_POST_unsuccessful_registration_no_password(self):
        register_data = {
            'email': 'test@test.com',
            'password': '',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        }
        response = self.client.post(self.register_page_url, register_data)
        self.assertFalse(User.objects.filter(email='test@test.com').exists())
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'user_register.html')

    def test_register_view_forbidden_method(self):
        response = self.client.put(self.register_page_url, data={})
        self.assertEqual(response.status_code, 403)

    def test_login_page_view_POST_unsuccessful_user_login(self):
        login_data = {
            "email": "test@pladat.com",
            "password": "WRONGpassword",
        }
        response = self.client.post(self.login_page_url, login_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "user_login.html")

    def test_register_user_valid_data_student(self):
        register_data = {
            'email': 'test@test.com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        }
        self.assertTrue(register_user(data=register_data))
        registered_user = User.objects.get(email='test@test.com')
        self.assertEqual(registered_user.email, 'test@test.com')
        registered_pladat_user = PladatUser.objects.get(user=registered_user)
        self.assertEqual(registered_pladat_user.user.email, 'test@test.com')
        self.assertEqual(registered_pladat_user.user_type, 0)

    def test_register_user_valid_data_recruiter(self):
        register_data = {
            'email': 'test@test.com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '1'
        }
        self.assertTrue(register_user(data=register_data))
        registered_user = User.objects.get(email='test@test.com')
        self.assertEqual(registered_user.email, 'test@test.com')
        registered_pladat_user = PladatUser.objects.get(user=registered_user)
        self.assertEqual(registered_pladat_user.user.email, 'test@test.com')
        self.assertEqual(registered_pladat_user.user_type, 1)
