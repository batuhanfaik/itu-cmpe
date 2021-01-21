from django.test import SimpleTestCase

from apps.pladat.forms import *


class TestRegistrationForm(SimpleTestCase):
    def test_registration_form_valid_data(self):
        form = RegistrationForm(data={
            'email':'test@test.com',
            'password':'password',
            'first_name':'unit',
            'last_name':'case',
            'phone_number':'+905555555555',
            'address':'my house is right on top of the world',
            'city':'aksaray',
            'state':'wat',
            'country':'Turkey',
            'user_type':'0',
        })
        self.assertTrue(form.is_valid())

    def test_registration_form_no_data(self):
        form = RegistrationForm(data={})

        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 9)

    def test_registration_no_email(self):
        form = LoginForm(data={
            'email': '',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_registration_invalid_email_no_domain(self):
        form = LoginForm(data={
            'email': 'test',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_registration_invalid_email_no_username(self):
        form = LoginForm(data={
            'email': '@test.com',
            'password': 'password',
            'first_name': 'unit',
            'last_name': 'case',
            'phone_number': '+905555555555',
            'address': 'my house is right on top of the world',
            'city': 'aksaray',
            'state': 'wat',
            'country': 'Turkey',
            'user_type': '0'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_registration_no_password(self):
        form = LoginForm(data={
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
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)



class TestLoginForm(SimpleTestCase):
    def test_login_form_valid_data(self):
        form = LoginForm(data={
            'email': 'test@test.com',
            'password': 'password'
        })
        self.assertTrue(form.is_valid())

    def test_login_form_no_data(self):
        form = LoginForm(data={})
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 2)

    def test_login_form_no_email(self):
        form = LoginForm(data={
            'password': 'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_login_form_no_password(self):
        form = LoginForm(data={
            'email': 'test@test.com'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_login_form_invalid_email_no_domain(self):
        form = LoginForm(data={
            'email': 'test',
            'password': 'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_login_form_invalid_email_no_username(self):
        form = LoginForm(data={
            'email': '@test.com',
            'password': 'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)
