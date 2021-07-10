from django.urls import path

from .views import *

urlpatterns = [
    path('<int:id>', job_view, name='job_view'),
    path('update/<int:id>', job_update_view, name='job_update_view'),
    path('list', job_list_view, name='job_list_view'),
    path('create', job_create_view, name='job_create_view'),
    path('<int:id>/find_student', find_student_view, name='find_student_view'),
    path('find_job', find_job_view, name='find_job_view'),
    path('<int:id>/matches', job_matches, name='job_matches'),
]
