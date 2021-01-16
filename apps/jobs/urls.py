from django.urls import path

from .views import *

urlpatterns = [
    path('applicant_profile', applicant_profile, name='applicant_profile')
]