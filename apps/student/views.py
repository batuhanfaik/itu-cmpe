from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User

from django.shortcuts import get_object_or_404

from apps.pladat.models import PladatUser
from .models import Student

from .forms import UpdatePladatUserForm, UpdateStudentForm

def profile_view(request, id):
    user = User.objects.filter(pk = id)

    if not user.exists():
        return HttpResponse('Profile does not exist')
    else:
        user = user[0]

    if user.pladatuser.user_type == PladatUser.UserType.COMPANY:
        return HttpResponse('Trying to look into a company page')

    try:
        user.pladatuser.student
    except:
        # Trying to look into their profile but did not completed it yet
        if request.user.id  == user.id:
            return redirect('/user/profile/update')
        else:
            return HttpResponse('Student did not complete the registration fully, missing information')

    return render(request, 'student_profile.html')

def profile_update_view(request):

    if request.method == 'GET':
        pladatuser = get_object_or_404(PladatUser, user = request.user)

        form1 = UpdatePladatUserForm(instance = pladatuser)
        
        student = get_object_or_404(Student, pladatuser = pladatuser)

        form2 = UpdateStudentForm(instance = student)

        ctx = {'form1': form1, 'form2': form2}

        if not request.user.is_authenticated:
            # User not logged in but trying to access profile update page
            # Redirect to the main page
            return redirect('/')

        return render(request, 'student_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform1' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        form1 = UpdatePladatUserForm(request.POST, instance = pladatuser)
        if form1.is_valid():
            form1.save()
            return HttpResponse('OK!')
        else:
            student = get_object_or_404(Student, pladatuser = pladatuser)
            form2 = UpdateStudentForm(instance = student)
            ctx = {'form1': form1, 'form2': form2}
            return render(request, 'student_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform2' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        student = get_object_or_404(Student, pladatuser = pladatuser)
        form2 = UpdateStudentForm(request.POST, instance = student)
        if form2.is_valid():
            form2.save()
            return HttpResponse('OK!')
        else:
            form1 = UpdatePladatUserForm(instance = pladatuser)
            ctx = {'form1': form1, 'form2': form2}
            return render(request, 'student_profile_update.html', context=ctx)

    return HttpResponse('What are you looking for?')