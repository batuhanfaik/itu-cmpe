from django.test import SimpleTestCase
from django.urls import resolve, reverse

from apps.recruiter.views import *

class TestUrls(SimpleTestCase):
    def setUp(self):
        pass
    def test_profile_update_view_url(self):
        url = reverse("profile_update_view")
        self.assertEqual(resolve(url).func, profile_update_view)