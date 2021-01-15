from django.urls import path

from .views import profile_update_view

urlpatterns = [
    path('profile/update', profile_update_view),
]