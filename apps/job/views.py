from django.shortcuts import render

from apps.pladat.models import PladatUser
from apps.pladat.models import User
from apps.job.models import Job
from apps.job.forms import UpdateJobForm
# Create your views here.


def job_update_view(request):
    job = Job.objects.get(id=1)
    user = User.objects.get(id=2)
    recruiter = user.pladatuser
    ctx = {
        'job':job,
        'recruiter':recruiter,
    }
    if request.method == 'GET':
        form = UpdateJobForm(instance=job)
        ctx['form'] = form
        return render(request, 'job_update.html', context=ctx)
    if request.method == 'POST':
        print('update jobs and shit')
        # TODO: add recruiter as a field if it is added to the model
        return render(request, 'job_update.html', context=ctx)

def job_view(request):
    job_application = Job.objects.get(id=1)
    recruiter = User.objects.get(id=2).pladatuser
    ctx = {
        'job':job_application,
        # TODO: remove recruiter if job contains a link to recruiter
        'recruiter':recruiter,
        'student_type':PladatUser.UserType.STUDENT,
    }
    if request.method == 'GET':
        return render(request, 'job.html', context=ctx)
    if request.method == 'POST':
        if 'yes' in request.POST:
            print('student interested in job')
        elif 'no' in request.POST:
            print('student is not interested in job')
        return render(request, 'job.html', context=ctx)


def applicant_profile(request):
    user = User.objects.get(email='test@pladat.com')
    applicant = PladatUser.objects.get(user=user)
    # TODO: check if student applied for this job!
    # TODO: check if job belongs to the recruiter(current user)
    ctx = {
        'applicant':applicant} #This should contain the PladatUser of the student
    if request.method == 'GET':
        print(ctx)
        return render(request, 'applicant_profile.html', context=ctx)
    if request.method == 'POST':
        if 'yes' in request.POST: # interested button is clicked
            print('yes')
        elif 'no' in request.POST: # not interested button is clicked
            print('no')
        return render(request, 'applicant_profile.html', context=ctx)