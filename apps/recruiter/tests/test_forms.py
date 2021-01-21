from django.test import SimpleTestCase

from apps.recruiter.forms import *
from apps.pladat.models import *


class TestUpdateRecruiterForm(SimpleTestCase):
    def test_valid_update_recruiter_form(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'google',
            'company_address': 'siligon valisi',
            'company_phone_number': '+905555555555',
        })
        self.assertTrue(form.is_valid())

    def test_update_recruiter_form_no_company_name(self):
        form = UpdateRecruiterForm(data={
            'company_name': '',
            'company_address': 'siligon valisi',
            'company_phone_number': '+905555555555',
        })
        self.assertFalse(form.is_valid())

    def test_update_recruiter_form_too_long_company_name(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'kajsbdfaksjfbaskdjfadkjfadfkjasdkasjfhasdkjhdhkfjadsbfkajsdkasjbfadkjfbadkfjabskdjasbkdfajdbkdajfbjakdfbdakfbjakdfbakdfbadkfjbadfadf',
            'company_address': 'siligon valisi',
            'company_phone_number': '+905555555555',
        })
        self.assertFalse(form.is_valid())

    def test_update_recruiter_form_no_company_address(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'google',
            'company_address': 'thisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtext',
            'company_phone_number': '+905555555555',
        })
        self.assertFalse(form.is_valid())

    def test_update_recruiter_form_no_phone_number(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'google',
            'company_address': 'siligon valisi',
            'company_phone_number': '',
        })
        self.assertFalse(form.is_valid())

    def test_update_recruiter_form_invalid_phone_number(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'google',
            'company_address': 'siligon valisi',
            'company_phone_number': 'sad',
        })
        self.assertFalse(form.is_valid())

    def test_update_recruiter_form_short_phone_number(self):
        form = UpdateRecruiterForm(data={
            'company_name': 'google',
            'company_address': 'siligon valisi',
            'company_phone_number': '+90555',
        })
        self.assertFalse(form.is_valid())
