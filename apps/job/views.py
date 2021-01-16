from django.shortcuts import render, redirect
from django.http import HttpResponse
from apps.pladat.models import PladatUser
from apps.pladat.models import User
from apps.job.models import Job, AppliedJob
from apps.recruiter.models import Recruiter
from apps.job.forms import UpdateJobForm
from django.shortcuts import get_object_or_404
# Create your views here.

def find_job_view(request):
    student = request.user.pladatuser.student
    pass

def job_find_student_view(request):
    if request.method != 'GET':
        return redirect('/')
    if not request.user.is_authenticated:
        return redirect('/')
    if request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
        return redirect('/')
    recruiter = request.user.pladatuser.recruiter
    pass

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
    if recruiter != job.recruiter:  # job is another recruiters job
        return redirect('/')  # TODO redirect where?
    ctx = {
        'job': job,
        'recruiter': recruiter,  # TODO it is unnecessary, job.recruiter exist
    }
    if request.method == 'GET':
        form = UpdateJobForm(instance=job)
        ctx['form'] = form
        return render(request, 'job_update.html', context=ctx)
    if request.method == 'POST':
        form = UpdateJobForm(request.POST, instance=job)
        ctx['form'] = form
        if form.is_valid():
            form.save()
            return redirect(f'/job/{id}')
        else:
            return render(request, 'job_update.html', context=ctx)
    return render(request, 'job_update.html', context=ctx)


def job_create_view(request):
    if not request.user.is_authenticated:
        return redirect('/')
    if request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
        return redirect('/')

    recruiter = request.user.pladatuser.recruiter

    ctx = {
        'recruiter': recruiter,
    }
    if request.method == 'GET':
        form = UpdateJobForm()
        ctx['form'] = form
        return render(request, 'job_update.html', context=ctx)
    if request.method == 'POST':
        form = UpdateJobForm(request.POST)
        ctx['form'] = form
        if form.is_valid():
            job = form.save(commit=False)
            job.recruiter = recruiter
            job.save()
            return redirect(f'/job/{job.id}')
        else:
            return render(request, 'job_update.html', context=ctx)
    return render(request, 'job_update.html', context=ctx)


def job_view(request, id):
    if not request.user.is_authenticated:
        return redirect('/')
    job = get_object_or_404(Job, id=id)
    ctx = {
        'job': job,
        'is_student': request.user.pladatuser.user_type == PladatUser.UserType.STUDENT,
        'match_rate': 100,  # TODO: pass match_rate
        'already_applied': False
    }
    if ctx["is_student"]:
        ctx['is_owner'] = False
        applied = AppliedJob.objects.filter(job=job, student=request.user.pladatuser.student)
        ctx['already_applied'] = bool(applied)
    else:
        ctx['is_owner'] = request.user.pladatuser.recruiter == job.recruiter

    if request.method == 'GET':
        return render(request, 'job.html', context=ctx)
    if request.method == 'POST' and ctx["is_student"]:
        application = AppliedJob(
            applicant=request.user.pladatuser.student,
            job=job,
        )
        if 'yes' in request.POST:
            application.is_student_interested = AppliedJob.StudentInterest.INTERESTED
        elif 'no' in request.POST:
            application.is_student_interested = AppliedJob.StudentInterest.NOT_INTERESTED
        application.save()
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
