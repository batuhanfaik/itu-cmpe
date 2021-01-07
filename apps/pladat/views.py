from django.shortcuts import render

from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render

from django.contrib.auth import authenticate, login

from .forms import RegistrationForm

from django.contrib.auth.models import User
from .models import PladatUser

def main_page_view(request):
    ctx = {}

    if not request.user.is_authenticated:
        # Check if user logged in
        ctx = {'user': 'guest'}

    return render(request, 'main_page.html', context=ctx)


def register_user(data):
    # Add a PladatUser
    user_dct = {
        'username': f'{data["first_name"]}{data["last_name"]}1231231',
        'email' : data['email'],
        'password': data['password'],
    }
    user = User.objects.create_user(**user_dct)
    user.save()

    fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country']
    pladatuser_dct = {key: data[key] for key in fields}
    pladatuser_dct['user'] = user

    pladatuser = PladatUser.objects.create(**pladatuser_dct)
    pladatuser.save()

    return pladatuser

def register_student(data):
    pladatuser = register_user(data)

    fields = ['degree', 'major', 'university', 'number_of_previous_work_experience', 'years_worked', 'is_currently_employed', 'skills_text']
    student_dct = {key: data[key] for key in fields}
    student_dct['pladatuser'] = pladatuser

    student = Student.objects.create(**student_dct)
    student.save()

    return student


def registration_view(request):
    ctx = {}
    
    if not request.user.is_authenticated:
        ctx['user'] = 'guest'
    else:
        return HttpResponse('Already a registered user')

    if request.method == 'GET':
        registration_form = RegistrationForm()
        ctx['form'] = registration_form
        return render(request, 'user_register.html', context = ctx)

    elif request.method == 'POST':
        registration_form = RegistrationForm(request.POST)
        if registration_form.is_valid():
            register_user(registration_form.data)
            # if registration_form.data['user_type'] == RegistrationForm.STUDENT:
            #     # Student
            #     register_student(registration_form.data)
            # else:
            #     # Company
            #     register_company(registration_form.data)
            return HttpResponse('Registered successfuly')
        else:
            ctx['form'] = registration_form
            print(registration_form.errors)
            return render(request, 'user_register.html', context = ctx)
    
    return HttpResponseForbidden()