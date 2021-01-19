from django.db import models
from django_countries.fields import CountryField
from phonenumber_field.modelfields import PhoneNumberField

from apps.pladat.models import PladatUser
from apps.recruiter.models import Recruiter
from apps.student.models import Student


class Response(models.IntegerChoices):
    NO_RESPONSE = 0, 'No response'
    INTERESTED = 1, 'Interested in this application'
    NOT_INTERESTED = 2, 'Not interested in this application'

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

    @property
    def company_name(self):
        return self.recruiter.company_name

    def appliedjob(self, student):
        appliedjob = AppliedJob.objects.filter(job = self, applicant = student)
        if len(appliedjob) > 0:
            return appliedjob[0]
        else:
            return None

    def is_applied(self, student):
        appliedjob = self.appliedjob(student)
        if appliedjob is None:
            return False
        if appliedjob.student_status == Response.NO_RESPONSE:
            return False
        return True


class AppliedJob(models.Model):
    applicant = models.ForeignKey(Student, on_delete=models.CASCADE, related_name = 'applications')
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name = 'applications')

    class Meta:
        unique_together = (("applicant", "job"))

    student_status  = models.IntegerField(choices=Response.choices, default=Response.NO_RESPONSE, help_text="Recruiter Response")
    recruiter_status = models.IntegerField(choices=Response.choices, default=Response.NO_RESPONSE, help_text="Recruiter Response")

    match_rate = models.IntegerField(null = True, help_text = 'Match rate')

    @property
    def is_student_interested(self):
        return self.student_status == Response.INTERESTED

    @property
    def is_student_no_response(self):
        return self.student_status == Response.NO_RESPONSE

    @property
    def is_recruiter_interested(self):
        return self.recruiter_status == Response.INTERESTED

    @property
    def is_recruiter_no_response(self):
        return self.recruiter_status == Response.NO_RESPONSE

class JobNotification(models.Model):
    appliedjob = models.ForeignKey(AppliedJob, on_delete=models.CASCADE, related_name='jobnotification')
    shown = models.BooleanField(default = False)