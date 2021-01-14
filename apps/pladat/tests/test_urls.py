from django.test import SimpleTestCase
from django.urls import reverse, resolve
from ..views import *


class TestUrls(SimpleTestCase):
    def setUp(self):
        pass

    def test_main_page_view(self):
        url = reverse("main_page_view")
        self.assertEqual(resolve(url).func, main_page_view)
