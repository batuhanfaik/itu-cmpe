from django.db import models

class Job(models.Model):
    title = models.CharField(max_length=128)
    description = models.TextField(max_length=512)
    requirements = models.TextField(max_length=512)
    city = models.CharField(max_length=128)
    state = models.CharField(max_length=128, null=True)
    country = CountryField()  # https://pypi.org/project/django-countries/