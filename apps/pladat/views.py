from django.shortcuts import render

from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, redirect

from django.contrib.auth import authenticate, login, logout

from .forms import RegistrationForm, LoginForm

from django.contrib.auth.models import User
from .models import PladatUser

from apps.student.models import Student
from apps.recruiter.models import Recruiter
from django.shortcuts import get_object_or_404

def main_page_view(request):
    ctx = {}
    return render(request, 'main_page.html', context=ctx)

def login_page_view(request):
    # TODO: Redirect back to the page it came from (Baris)
    # TODO: Does not show errors when wrong email / password (Baris)
    # The page it came from can be reached by request.REQUEST.get('next')
    ctx = {}
    if request.method == 'GET':
        if request.user.is_authenticated:
            return HttpResponse('''
                Already logged in
                <script>
                    function redirect(){
                    window.location.href = "/";
                    }
                    setTimeout(redirect, 1000);
                </script>
            ''')
        login_form = LoginForm()
        ctx = {'form': login_form}
        return render(request, 'user_login.html', context=ctx)
    elif request.method == 'POST':
        login_form = LoginForm(request.POST)
        if login_form.is_valid():
            email = login_form.data['email']
            password = login_form.data['password']
            user = authenticate(username=email, password=password)
            if user is not None:
                login(request, user)
                return redirect(main_page_view)
            else:
                login_form.add_error(field="email", error='Not user found with given email and password')
                ctx['form'] = login_form
                return render(request, 'user_login.html', context=ctx)
        else:
            ctx['form'] = login_form
            return render(request, 'user_login.html', context=ctx)
    else:
        return HttpResponseForbidden('Forbidden method')


def register_user(data):
    # TODO: Add email validation (Baris)
    try:
        user_dct = {
            'username': data['email'],
            'email': data['email'],
            'password': data['password'],
        }
        user = User.objects.create_user(**user_dct)

        fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country',
                'user_type']
        pladatuser_dct = {key: data[key] for key in fields}
        pladatuser_dct['user'] = user

        pladatuser = PladatUser.objects.create(**pladatuser_dct)

        if int(data['user_type']) == PladatUser.UserType.STUDENT:
            student_dct = {'pladatuser': pladatuser}
            Student.objects.create(**student_dct)

        elif int(data['user_type']) == PladatUser.UserType.RECRUITER:
            recruiter_dct = {'pladatuser': pladatuser}
            Recruiter.objects.create(**recruiter_dct)
        return True
    except:
        return False


def registration_view(request):
    ctx = {}

    if request.user.is_authenticated:
        # This happens when an logged in user visits the register page
        # We might consider redirecting here instead
        # return HttpResponse('Already a registered user')
        return redirect('/')

    if request.method == 'GET':
        # Unregistered user trying to access the registration form
        registration_form = RegistrationForm()
        ctx['form'] = registration_form
        return render(request, 'user_register.html', context=ctx)

    elif request.method == 'POST':
        # Form filled and submitted
        registration_form = RegistrationForm(request.POST)
        if registration_form.is_valid():
            email = registration_form.data['email']
            if User.objects.filter(email=email).exists():
                # Email already in use
                registration_form.add_error('email', 'Email is already in use')
                ctx['form'] = registration_form
                return render(request, 'user_register.html', context=ctx)
            else:
                register_user(registration_form.data)
                return HttpResponse('''
                Registered successfully
                <script>
                    function redirect(){
                    window.location.href = "/";
                    }
                    setTimeout(redirect, 1000);
                </script>
                    ''')
        else:
            # Something wrong with form
            ctx['form'] = registration_form
            return render(request, 'user_register.html', context=ctx)

    return HttpResponseForbidden()


def logout_page_view(request):
    if request.user.is_authenticated:
        logout(request)
    return redirect('/')


def profile_view(request, id):
    user = get_object_or_404(User, id=id)

    ctx = {
        'owner': id == request.user.id,
        'profile_user': user,
    }

    if user.pladatuser.is_student():
        return render(request, 'student_profile.html', context=ctx)
    else:
        return render(request, 'recruiter_profile.html', context=ctx)
