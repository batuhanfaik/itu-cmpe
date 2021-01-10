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
        'username': data['email'],
        'email' : data['email'],
        'password': data['password'],
    }
    user = User.objects.create_user(**user_dct)
    user.save()

    fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country', 'user_type']
    pladatuser_dct = {key: data[key] for key in fields}
    pladatuser_dct['user'] = user

    pladatuser = PladatUser.objects.create(**pladatuser_dct)
    pladatuser.save()

    if data['user_type'] == PladatUser.User_Type.STUDENT:
        fields = ['degree', 'major', 'university', 'number_of_previous_work_experience', 'years_worked', 'is_currently_employed', 'skills_text']
        student_dct = {key: data[key] for key in fields}
        student_dct['pladatuser'] = pladatuser

        student = Student.objects.create(**student_dct)
        student.save()
        
    elif data['user_type'] == PladatUser.User_Type.COMPANY:
        recruiter = Recruiter.objects.create(pladatuser = pladatuser)
        recruiter.save()

    return True


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
            email = registration_form.data['email']

            if User.objects.filter(email = email).exists():
                # Email already in use
                registration_form.add_error('email', 'Email is already in use')
                ctx['form'] = registration_form
                return render(request, 'user_register.html', context = ctx)

            register_user(registration_form.data)
            return HttpResponse('Registered successfuly')
        else:
            # Something wrong with form
            ctx['form'] = registration_form
            return render(request, 'user_register.html', context = ctx)
    
    return HttpResponseForbidden()


def student_profile_view(request):
    ctx = {}

    # if not request.user.is_authenticated:
    #     # Check if user logged in
    #     ctx = {'user': 'guest'}

    return render(request, 'student_profile.html', context=ctx)

def student_profile_update_view(request):
    ctx = {}

    # if not request.user.is_authenticated:
    #     # Check if user logged in
    #     ctx = {'user': 'guest'}

    return render(request, 'student_profile_update.html', context=ctx)