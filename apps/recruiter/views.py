from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from apps.pladat.models import PladatUser

from .forms import UpdateRecruiterForm
from .models import Recruiter
from apps.pladat.forms import UpdateImageForm, UpdatePladatUserForm


def profile_update_view(request):
    if request.method == 'GET':
        if not request.user.is_authenticated \
                or request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
            # User not logged in but trying to access profile update page
            # Redirect to the main page
            return redirect('/')

        pladatuser = request.user.pladatuser
        form1 = UpdatePladatUserForm(instance=pladatuser)
        recruiter = pladatuser.recruiter
        form2 = UpdateRecruiterForm(instance=recruiter)
        imageForm = UpdateImageForm()

        ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}

        return render(request, 'recruiter_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform1' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user=request.user)
        form1 = UpdatePladatUserForm(request.POST, instance=pladatuser)
        imageForm = UpdateImageForm()
        if form1.is_valid():
            form1.save()
            return redirect('/profile/' + str(request.user.id))
        else:
            recruiter = get_object_or_404(Recruiter, pladatuser=pladatuser)
            form2 = UpdateRecruiterForm(instance=recruiter)
            ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
            return render(request, 'recruiter_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform2' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user=request.user)
        recruiter = get_object_or_404(Recruiter, pladatuser=pladatuser)
        form2 = UpdateRecruiterForm(request.POST, instance=recruiter)
        imageForm = UpdateImageForm()
        if form2.is_valid():
            form2.save()
            return redirect('/profile/' + str(request.user.id))
        else:
            form1 = UpdatePladatUserForm(instance=pladatuser)
            ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
            return render(request, 'recruiter_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnimageform' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user=request.user)
        recruiter = get_object_or_404(Recruiter, pladatuser=pladatuser)
        imageForm = UpdateImageForm(request.POST, request.FILES)
        if imageForm.is_valid():
            img = imageForm.cleaned_data.get("image")
            pladatuser.image = img
            pladatuser.save()
            return redirect('/profile/' + str(request.user.id))
        else:
            form1 = UpdatePladatUserForm(instance=pladatuser)
            form2 = UpdateRecruiterForm(instance=recruiter)
            ctx = {'form1': form1, 'form2': form2, 'imageForm': imageForm}
            return render(request, 'student_profile_update.html', context=ctx)

    return HttpResponse('What are you looking for?')
