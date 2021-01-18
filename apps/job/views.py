from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render

from apps.job.forms import UpdateJobForm
from apps.job.models import AppliedJob, Job
from apps.pladat.models import PladatUser
from apps.recruiter.models import Recruiter
from apps.student.models import Student

from .models import Response

# Create your views here.


def find_job_view(request):
    student = request.user.pladatuser.student
    pass


def calculate_match_rate(student, job):
    # TODO: Add this after ML (Baris)
    return 100


def find_students(job, index):
    student_list = [
        (x.applicant, calculate_match_rate(x.applicant, job))
        for x in AppliedJob.objects.filter(
            job=job,
            student_status=Response.INTERESTED,
            recruiter_status=Response.NO_RESPONSE,
        )
    ]


    if len(student_list) == 0:
        return None

    # Get list of students
    def compare(student1, student2):
        score1 = student1[1]
        score2 = student2[1]
        if score1 < score2:
            return -1
        elif score1 > score2:
            return 1
        else:
            return 0

    from functools import cmp_to_key

    sorted(student_list, key=cmp_to_key(compare))

    return student_list[index % len(student_list)]

@login_required
def job_find_student_view(request, id):
    if request.method == "GET":

        if request.user.pladatuser.is_student():
            return HttpResponseForbidden("Invalid user")

        index = request.GET.get('index', 0)

        recruiter = request.user.pladatuser.recruiter

        job = get_object_or_404(Job, pk=id)

        student = find_students(job, index)

        ctx = {"job": job, "student": student[0], "match_rate": student[1]}

        return render(request, 'find_student.html', context = ctx)

    else:
        HttpResponseForbidden("Forbidden method")


@login_required
def job_list_view(request):
    if request.method == "GET":
        if request.user.pladatuser.is_student():
            return HttpResponseForbidden("Invalid user")
        recruiter = request.user.pladatuser.recruiter
        job_list = Job.objects.filter(recruiter=recruiter)
        ctx = {
            "job_list": job_list,
        }
        return render(request, "job_list.html", context=ctx)

    return HttpResponseForbidden("Forbidden method")


@login_required
def job_update_view(request, id):
    if request.user.pladatuser.is_student():
        return redirect("/")
    job = get_object_or_404(Job, id=id)
    recruiter = request.user.pladatuser.recruiter
    if recruiter != job.recruiter:  # job is another recruiters job
        return redirect("/")  # TODO redirect where?
    ctx = {
        "job": job,
        "recruiter": recruiter,  # TODO it is unnecessary, job.recruiter exist
    }
    if request.method == "GET":
        form = UpdateJobForm(instance=job)
        ctx["form"] = form
        return render(request, "job_update.html", context=ctx)
    if request.method == "POST":
        form = UpdateJobForm(request.POST, instance=job)
        ctx["form"] = form
        if form.is_valid():
            form.save()
            return redirect(f"/job/{id}")
        else:
            return render(request, "job_update.html", context=ctx)
    return render(request, "job_update.html", context=ctx)


@login_required
def job_create_view(request):
    if request.user.pladatuser.is_student():
        return redirect("/")

    recruiter = request.user.pladatuser.recruiter
    ctx = {
        "recruiter": recruiter,
    }
    if request.method == "GET":
        form = UpdateJobForm()
        ctx["form"] = form
        return render(request, "job_update.html", context=ctx)

    if request.method == "POST":
        form = UpdateJobForm(request.POST)
        ctx["form"] = form
        if form.is_valid():
            job = form.save(commit=False)
            job.recruiter = recruiter
            job.save()
            return redirect(f"/job/{job.id}")
        else:
            return render(request, "job_update.html", context=ctx)

    return render(request, "job_update.html", context=ctx)


@login_required
def job_view(request, id):
    job = get_object_or_404(Job, id=id)

    ctx = {
        "job": job,
        "is_student": request.user.pladatuser.is_student(),
    }

    if ctx["is_student"]:
        student = request.user.pladatuser.student

        ctx["is_owner"] = False

        appliedjob = job.appliedjob(student)
        ctx["appliedjob"] = appliedjob

        ctx["match_rate"] = calculate_match_rate(student, job)

    else:
        ctx["is_owner"] = request.user.pladatuser.recruiter == job.recruiter

    if request.method == "GET":
        return render(request, "job.html", context=ctx)

    if request.method == "POST" and ctx["is_student"]:

        appliedjob = job.appliedjob(student)
        if appliedjob:
            return HttpResponseForbidden("Invalid request")

        application = AppliedJob(
            applicant=request.user.pladatuser.student,
            job=job,
        )
        if "yes" in request.POST:
            application.student_status = Response.INTERESTED

        elif "no" in request.POST:
            application.student_status = Response.NOT_INTERESTED

        application.save()

        ctx["appliedjob"] = application

        return render(request, "job.html", context=ctx)

    return HttpResponseForbidden("Forbidden method")


