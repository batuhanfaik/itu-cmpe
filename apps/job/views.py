from django.shortcuts import render, redirect
from django.http import HttpResponse
from apps.pladat.models import PladatUser
from apps.pladat.models import User
from apps.job.models import Job
from apps.recruiter.models import Recruiter
from apps.job.forms import UpdateJobForm
from django.shortcuts import get_object_or_404
# Create your views here.


def job_list_view(request):
    if request.method != 'GET':
        return redirect('/')
    if not request.user.is_authenticated:
        return redirect('/')
    if request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
        return redirect('/')
    recruiter = request.user.pladatuser.recruiter
    job_list = Job.objects.filter(recruiter=recruiter)
    ctx = {
        'job_list': job_list,
    }
    return render(request, 'job_list.html', context=ctx)


def job_update_view(request, id):
    if not request.user.is_authenticated:
        return redirect('/')
    if request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
        return redirect('/')
    job = get_object_or_404(Job, id=id)
    recruiter = request.user.pladatuser.recruiter
    if recruiter != job.recruiter: # job is another recruiters job
        return redirect('/') # TODO redirect where?
    ctx = {
        'job': job,
        'recruiter': recruiter,
    }
    if request.method == 'GET':
        form = UpdateJobForm(instance=job)
        ctx['form'] = form
        return render(request, 'job_update.html', context=ctx)
    if request.method == 'POST':
        form = UpdateJobForm(request.POST, instance=job)
        if form.is_valid():
            form.save()
            return redirect(f'/job/{id}')
        return render(request, 'job_update.html', context=ctx)
    return render(request, 'job_update.html', context=ctx)

def job_view(request, id):
    if not request.user.is_authenticated:
        return redirect('/')
    job = get_object_or_404(Job, id=id)
    ctx = {
        'job': job,
        'is_student': request.user.pladatuser.user_type == PladatUser.UserType.STUDENT,
    }
    if ctx["is_student"]:
        ctx['is_owner'] = False
    else:
        ctx['is_owner'] = request.user.pladatuser.recruiter == job.recruiter

    if request.method == 'GET':
        return render(request, 'job.html', context=ctx)
    if request.method == 'POST' and ctx["is_student"]:
        if 'yes' in request.POST:
            print('student interested in job')
        elif 'no' in request.POST:
            print('student is not interested in job')
        # TODO send next job
        return render(request, 'job.html', context=ctx)

    return HttpResponse("You should not be here")


# TODO i did not look here yet _osman
def applicant_profile(request):
    user = User.objects.get(email='test@pladat.com')
    applicant = PladatUser.objects.get(user=user)
    # TODO: check if student applied for this job!
    # TODO: check if job belongs to the recruiter(current user)
    ctx = {
        'applicant': applicant}  # This should contain the PladatUser of the student
    if request.method == 'GET':
        print(ctx)
        return render(request, 'applicant_profile.html', context=ctx)
    if request.method == 'POST':
        if 'yes' in request.POST:  # interested button is clicked
            print('yes')
        elif 'no' in request.POST:  # not interested button is clicked
            print('no')
        return render(request, 'applicant_profile.html', context=ctx)
