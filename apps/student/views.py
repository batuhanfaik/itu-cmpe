from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from apps.pladat.models import PladatUser

from .forms import UpdateStudentForm
from .models import Student
from apps.pladat.forms import UpdateImageForm, UpdatePladatUserForm

@login_required
def profile_update_view(request):
    if request.method == 'GET':
        pladatuser = get_object_or_404(PladatUser, user=request.user)
        form1 = UpdatePladatUserForm(instance = pladatuser)
        student = get_object_or_404(Student, pladatuser=pladatuser)
        form2 = UpdateStudentForm(instance=student)
        imageForm = UpdateImageForm()

        ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}

        return render(request, 'student_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform1' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        student = get_object_or_404(Student, pladatuser=pladatuser)
        form1 = UpdatePladatUserForm(request.POST, instance = pladatuser)
        if form1.is_valid():
            form1.save()
            return redirect('/profile/' + str(request.user.id))
        else:
            form2 = UpdateStudentForm(instance = student)
            imageForm = UpdateImageForm(request.POST)
            ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
            return render(request, 'student_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform2' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        student = get_object_or_404(Student, pladatuser = pladatuser)
        form2 = UpdateStudentForm(request.POST, instance = student)
        if form2.is_valid():
            form2.save()
            return redirect('/profile/' + str(request.user.id))
        else:
            form1 = UpdatePladatUserForm(instance = pladatuser)
            imageForm = UpdateImageForm(request.POST)
            ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
            return render(request, 'student_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnimageform' in request.POST:
            pladatuser = get_object_or_404(PladatUser, user=request.user)
            student = get_object_or_404(Student, pladatuser=pladatuser)
            imageForm = UpdateImageForm(request.POST, request.FILES)
            if imageForm.is_valid():
                img = imageForm.cleaned_data.get("image")
                pladatuser.image = img
                pladatuser.save()
                return redirect('/profile/' + str(request.user.id))
            else:
                form1 = UpdatePladatUserForm(instance=pladatuser)
                form2 = UpdateStudentForm(instance=student)
                ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
                return render(request, 'student_profile_update.html', context=ctx)

    return HttpResponse('What are you looking for?')