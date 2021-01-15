from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User

from django.shortcuts import get_object_or_404

from apps.pladat.models import PladatUser
from .models import Recruiter

from .forms import UpdatePladatUserForm, UpdateRecruiterForm


def profile_update_view(request):

    if request.method == 'GET':
        pladatuser = get_object_or_404(PladatUser, user=request.user)
        form1 = UpdatePladatUserForm(instance=pladatuser)
        recruiter = get_object_or_404(Recruiter, pladatuser=pladatuser)
        form2 = UpdateRecruiterForm(instance=recruiter)

        ctx = {'form1': form1, 'form2': form2}

        if not request.user.is_authenticated:
            # User not logged in but trying to access profile update page
            # Redirect to the main page
            return redirect('/')

        return render(request, 'recruiter_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform1' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        form1 = UpdatePladatUserForm(request.POST, instance = pladatuser)
        if form1.is_valid():
            form1.save()
            return redirect('/recruiter/profile/' + str(request.user.id))
        else:
            recruiter = get_object_or_404(Recruiter, pladatuser = pladatuser)
            form2 = UpdateRecruiterForm(instance = recruiter)
            ctx = {'form1': form1, 'form2': form2}
            return render(request, 'recruiter_profile_update.html', context=ctx)

    if request.method == 'POST' and 'btnform2' in request.POST:
        pladatuser = get_object_or_404(PladatUser, user = request.user)
        recruiter = get_object_or_404(Recruiter, pladatuser = pladatuser)
        form2 = UpdateRecruiterForm(request.POST, instance = recruiter)
        if form2.is_valid():
            form2.save()
            return redirect('/recruiter/profile/' + str(request.user.id))
        else:
            form1 = UpdatePladatUserForm(instance = pladatuser)
            ctx = {'form1': form1, 'form2': form2}
            return render(request, 'recruiter_profile_update.html', context=ctx)

    return HttpResponse('What are you looking for?')