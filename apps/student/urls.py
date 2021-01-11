from django.urls import path

from .views import profile_view, profile_update_view

urlpatterns = [
    path('profile/<int:id>', profile_view),
    path('update', profile_update_view),
]