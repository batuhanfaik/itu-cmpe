from django.urls import path

from .views import (login_page_view, logout_page_view, main_page_view,
                    profile_view, registration_view)

urlpatterns = [
    path('', main_page_view, name="main_page_view"),
    path('register', registration_view, name="registration_view"),
    path('login', login_page_view, name="login_page_view"),
    path('logout', logout_page_view, name="logout_page_view"),
    path('profile/<int:id>', profile_view, name="profile_view"),
]