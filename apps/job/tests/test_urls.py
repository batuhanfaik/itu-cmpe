from django.test import SimpleTestCase
from django.urls import resolve, reverse

from apps.job.views import *


class TestUrls(SimpleTestCase):
    def setUp(self):
        pass

    def test_job_view(self):
        url = reverse("job_view", kwargs={'id': 1})
        self.assertEqual(resolve(url).func, job_view)

    def test_job_update_view(self):
        url = reverse("job_update_view", kwargs={'id': 1})
        self.assertEqual(resolve(url).func, job_update_view)

    def test_job_list_view(self):
        url = reverse("job_list_view")
        self.assertEqual(resolve(url).func, job_list_view)

    def test_job_create_view(self):
        url = reverse("job_create_view")
        self.assertEqual(resolve(url).func, job_create_view)

    def test_find_student_view(self):
        url = reverse("find_student_view", kwargs={'id': 1})
        self.assertEqual(resolve(url).func, find_student_view)

    def test_find_job_view(self):
        url = reverse("find_job_view")
        self.assertEqual(resolve(url).func, find_job_view)

    def test_job_matches(self):
        url = reverse("job_matches", kwargs={'id': 1})
        self.assertEqual(resolve(url).func, job_matches)
