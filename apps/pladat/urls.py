from django.urls import path

from .views import main_page_view, registration_view

urlpatterns = [
    path('', main_page_view),
    path('register', registration_view)
]