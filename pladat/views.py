from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render

from .forms import RegistrationForm

import json

def user_register_view(request):
    if request.method == 'GET':
        # Create an empty form
        registration_form = RegistrationForm()
        return render(request, 'user_register.html', context={'form': registration_form})
    elif request.method == 'POST':
        # Get filled form data
        registration_form = RegistrationForm(request.POST)
        print(registration_form.data)
        if registration_form.is_valid():
            return HttpResponse('Registered successfuly')
        else:
            return HttpResponse('Registration failed')