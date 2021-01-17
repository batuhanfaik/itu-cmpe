from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseForbidden
from apps.pladat.models import PladatUser
from apps.pladat.models import User
from apps.job.models import Job, AppliedJob
from apps.recruiter.models import Recruiter
from apps.job.forms import UpdateJobForm
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
# Create your views here.

def find_job_view(request):
    student = request.user.pladatuser.student
    pass

@login_required
def job_find_student_view(request):
    if request.method == 'GET':
        # Student can not access this page
        if request.user.pladatuser.is_student():
            return HttpResponseForbidden('Invalid user')
        recruiter = request.user.pladatuser.recruiter
        return HttpResponse(recruiter)
    else:
        HttpResponseForbidden('Forbidden method')

@login_required
def job_list_view(request):
    if request.method != 'GET':
        return redirect('/')
    if request.user.pladatuser.user_type == PladatUser.UserType.STUDENT:
        return redirect('/')
    recruiter = request.user.pladatuser.recruiter
    job_list = Job.objects.filter(recruiter=recruiter)
    ctx = {
        'job_list': job_list,
    }
    return render(request, 'job_list.html', context=ctx)

@login_required
def job_update_view(request, id):
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

@login_required
def job_create_view(request):
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

def calculate_match_rate(student, job):
    # TODO: Add this after ML (Baris)
    return 100

@login_required
def job_view(request, id):
    job = get_object_or_404(Job, id=id)
    
    ctx = {
        'job': job,
        'is_student': request.user.pladatuser.is_student(),
        'already_applied': False
    }
    
    if ctx["is_student"]:
        student = request.user.pladatuser.student
        ctx['is_owner'] = False
        ctx['already_applied'] = job.is_applied(student)
        ctx['match_rate'] = calculate_match_rate(student, job)

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

@login_required
def applicant_profile(request, applied_job_id):
    application = get_object_or_404(AppliedJob, applied_job_id)

    if request.method == 'GET':
        applicant_pladatuser = application.applicant.pladatuser
        ctx = {'applicant': applicant_pladatuser}
        return render(request, 'applicant_profile.html', context=ctx)
    
    if request.method == 'POST':
        recruiter_response = None
        
        if 'yes' in request.POST:  # interested button is clicked
            recruiter_response = AppliedJob.RecruiterResponse.INTERESTED
        
        elif 'no' in request.POST:  # not interested button is clicked
            recruiter_response = AppliedJob.RecruiterResponse.NOT_INTERESTED
        
        appliedjob.recruiter_response = recruiter_response
        appliedjob.save()

        return render(request, 'applicant_profile.html', context=ctx)
    
    return HttpResponseForbidden('Forbidden method')