from django.urls import path

from .views import (main_page_view,
    registration_view,
    login_page_view,
    recruiter_profile_view,
    recruiter_profile_update_view,
    logout_page_view)

urlpatterns = [
    path('', main_page_view, name="main_page_view"),
    path('register', registration_view, name="registration_view"),
    path('login', login_page_view, name="login_page_view"),
    path('logout', logout_page_view, name="logout_page_view"),
    path('recruiter_profile', recruiter_profile_view, name="recruiter_profile_view"),
    path('recruiter_profile_update', recruiter_profile_update_view, name="recruiter_profile_update_view")
]