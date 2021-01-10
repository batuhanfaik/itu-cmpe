from django.urls import path

from .views import main_page_view, registration_view, student_profile_view, student_profile_update_view

urlpatterns = [
    path('', main_page_view),
    path('register', registration_view),
    path('student_profile', student_profile_view),
    path('student_profile_update', student_profile_update_view)
]