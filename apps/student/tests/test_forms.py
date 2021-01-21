from django.test import TestCase

from apps.student.forms import *
from apps.student.models import *


class TestUpdateStudentForm(TestCase):
    def setUp(self):
        self.test_skill = Skill.objects.create(
            name="Skill1"
        )

    def test_valid_update_student_form(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertTrue(form.is_valid())

    def test_update_student_form_no_skill(self):
        form = UpdateStudentForm(data={
            'skills': '',
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_text_skill(self):
        form = UpdateStudentForm(data={
            'skills': 'Skill1',
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_no_degree(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': '',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_no_major(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': '',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_no_university(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'cmpe',
            'university': '',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_negative_working_years(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': -1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_zero_working_years(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 0,
        })
        self.assertTrue(form.is_valid())

    def test_update_student_form_too_long_degree(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'thisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtext',
            'major': 'cmpe',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_too_long_major(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'thisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtext',
            'university': 'itu',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())

    def test_update_student_form_too_long_university(self):
        form = UpdateStudentForm(data={
            'skills': [self.test_skill],
            'degree': 'bsc',
            'major': 'cmpe',
            'university': 'thisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtextthisisaverylongtext',
            'years_worked': 1,
        })
        self.assertFalse(form.is_valid())
