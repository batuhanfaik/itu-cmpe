from django.db import models

from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser
from apps.recruiter.models import Recruiter
from apps.student.models import Student


class Job(models.Model):
    title = models.CharField(max_length=128, help_text='Title')
    description = models.TextField(max_length=512, help_text='Description')
    requirements = models.TextField(max_length=512, help_text='Requirements')
    city = models.CharField(max_length=128, help_text='City')
    state = models.CharField(max_length=128, null=True, help_text='State')
    country = CountryField(help_text='Country')  # https://pypi.org/project/django-countries/
    recruiter = models.ForeignKey(Recruiter, on_delete=models.CASCADE)
    # Company info also required buy can be derived from recruiter
    def __str__(self):
        return self.title


class AppliedJob(models.Model):
    applicant = models.ForeignKey(Student, on_delete=models.CASCADE)
    job = models.ForeignKey(Job, on_delete=models.CASCADE)

    class Meta:
        unique_together = (("applicant", "job"))
    class StudentInterest(models.IntegerChoices):
        INTERESTED = 1, 'Student is interested with the job'
        NOT_INTERESTED = 0, 'Student is not interested with the job'
    is_student_interested  = models.IntegerField(choices=StudentInterest.choices, help_text="Recruiter Response")

    class RecruiterResponse(models.IntegerChoices):
        NO_RESPONSE = 0, 'No response from recruiter'
        INTERESTED = 1, 'Recruiter is interested with this student'
        NOT_INTERESTED = 2, 'Recruiter is not interested with this student'
    recruiter_response = models.IntegerField(choices=RecruiterResponse.choices, default=RecruiterResponse.NO_RESPONSE, help_text="Recruiter Response")
    