from django.test import SimpleTestCase
from apps.pladat.forms import *

class TestForms(SimpleTestCase):
    def test_login_form_valid_data(self):
        form = LoginForm(data={
            'email':'test@test.com',
            'password':'password'
        })
        self.assertTrue(form.is_valid())

    def test_login_form_no_data(self):
        form = LoginForm(data={})
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors),2)

    def test_login_form_no_email(self):
        form = LoginForm(data={
            'password':'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_login_form_no_password(self):
        form = LoginForm(data={
            'email':'test@test.com'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)

    def test_login_form_invalid_email_no_domain(self):
        form = LoginForm(data={
            'email':'test',
            'password':'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors),1)

    def test_login_form_invalid_email_no_username(self):
        form = LoginForm(data={
            'email':'@test.com',
            'password':'password'
        })
        self.assertFalse(form.is_valid())
        self.assertEqual(len(form.errors), 1)