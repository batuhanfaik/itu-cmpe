from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User

from apps.pladat.models import PladatUser
from .models import Student


def profile_view(request, id):
    user = User.objects.filter(pk = id)

    if not user.exists():
        return HttpResponse('Profile does not exist')
    else:
        user = user[0]

    if user.pladatuser.user_type == PladatUser.UserType.COMPANY:
        return HttpResponse('Trying to look into a company page')
    
    # Send the pladatuser object to the template
    ctx = {'user': user}

    return render(request, 'student_profile.html', context=ctx)

def profile_update_view(request):
    ctx = {}

    if request.user.is_authenticated:
        ctx = {'user': request.user}
    else:
        # User not logged in but trying to access profile update page
        # Redirect to the main page
        return redirect('/')

    return render(request, 'student_profile_update.html', context=ctx)