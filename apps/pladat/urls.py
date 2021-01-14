from django.urls import path

from .views import main_page_view, registration_view, login_page_view

urlpatterns = [
    path('', main_page_view),
    path('register', registration_view),
    path('login', login_page_view),
]