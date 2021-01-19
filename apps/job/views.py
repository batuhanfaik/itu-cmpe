from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render

from apps.job.forms import UpdateJobForm
from apps.job.models import AppliedJob, Job, JobNotification
from apps.pladat.models import PladatUser
from apps.recruiter.models import Recruiter
from apps.student.models import Student

from .models import Response

from apps.recommend.rec_utils import create_features 
from apps.recommend.rec_model import predict


def calculate_match_rate(student, job):
    query = create_features(student, job)

    return predict(query)


def match_rate(student, job):
    appliedjob = AppliedJob.objects.filter(job=job, applicant=student)

    if len(appliedjob) is 0:
        return calculate_match_rate(student, job)

    appliedjob = appliedjob[0]

    if appliedjob.match_rate is None:
        appliedjob.match_rate = calculate_match_rate(student, job)
        appliedjob.save()
    return appliedjob.match_rate


def find_student(job, index):
    student_list = [
        (x.applicant, match_rate(x.applicant, job))
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
            return 1
        elif score1 > score2:
            return -1
        else:
            return 0

    from functools import cmp_to_key

    sorted(student_list, key=cmp_to_key(compare))

    return student_list[index % len(student_list)]


def find_job(student, index):
    job_list = [x for x in Job.objects.all()]

    job_list = list(filter(lambda job: not job.is_applied(student), job_list))

    if len(job_list) == 0:
        return None

    job_list_scored = [(x, match_rate(student, x)) for x in job_list]

    # Get list of students
    def compare(scored_job1, scored_job2):
        score1 = scored_job1[1]
        score2 = scored_job2[1]
        if score1 < score2:
            return 1
        elif score1 > score2:
            return -1
        else:
            return 0

    from functools import cmp_to_key

    sorted(job_list_scored, key=cmp_to_key(compare))

    return job_list_scored[index % len(job_list_scored)]


@login_required
def find_student_view(request, id):

    if request.user.pladatuser.is_student():
        return HttpResponseForbidden("Invalid user")

    if request.method == "GET":
        index = int(request.GET.get("index", "0"))
        job = get_object_or_404(Job, pk=id)
        recruiter = request.user.pladatuser.recruiter
        if recruiter != job.recruiter:
            return HttpResponseForbidden("Invalid User")  # TODO

        student = find_student(job, index)
        if student is None:
            # TODO: Return some page...
            return redirect('/job/list')

        ctx = {"job": job, "student": student[0], "match_rate": student[1]}
        return render(request, "find_student.html", context=ctx)

    if request.method == "POST":
        data = request.POST

        index = int(data.get("index", "0"))

        job = get_object_or_404(Job, pk=id)
        recruiter = request.user.pladatuser.recruiter
        if recruiter != job.recruiter:
            return HttpResponseForbidden("Invalid user")  # TODO

        studentid = int(data["studentid"])
        pladatuser = get_object_or_404(PladatUser, pk=studentid)

        student = pladatuser.student

        # Check if applied before
        appliedjob = AppliedJob.objects.filter(job=job, applicant=student)

        if len(appliedjob) == 0:
            return HttpResponseForbidden("Invalid request")

        appliedjob = appliedjob[0]

        if not appliedjob.is_recruiter_no_response:
            return HttpResponseForbidden("Invalid request")

        if "yes" in request.POST:
            appliedjob.recruiter_status = Response.INTERESTED


        elif "no" in request.POST:
            appliedjob.recruiter_status = Response.NOT_INTERESTED

        appliedjob.save()

        return redirect(f"/job/{id}/find_student?index={index}")

    return HttpResponseForbidden("Forbidden method")


@login_required
def find_job_view(request):

    if not request.user.pladatuser.is_student():
        return HttpResponseForbidden("Invalid user")

    ctx = {}
    ctx["is_owner"] = False
    ctx["is_student"] = True

    if request.method == "GET":
        index = int(request.GET.get("index", "0"))

        student = request.user.pladatuser.student
        job = find_job(student, index)

        if job is None:
            # TODO: Return some page...
            return HttpResponse("No job found")

        ctx["job"] = job[0]
        ctx["match_rate"] = job[1]

        return render(request, "job.html", context=ctx)

    if request.method == "POST":

        data = request.POST
        index = int(data.get("index", "0"))
        jobid = int(request.POST["jobid"])
        job = get_object_or_404(Job, id=jobid)
        student = request.user.pladatuser.student

        # Check if applied before
        appliedjob = job.appliedjob(student)

        if appliedjob:
            return HttpResponseForbidden("Invalid request")

        application = AppliedJob(
            applicant=student,
            job=job,
        )
        if "yes" in request.POST:
            application.student_status = Response.INTERESTED
            application.save()
            JobNotification.objects.create(appliedjob=application)

        elif "no" in request.POST:
            application.student_status = Response.NOT_INTERESTED
            application.save()

        return redirect(f"/job/find_job?index={index}")

    else:
        HttpResponseForbidden("Forbidden method")


@login_required
def job_matches(request, id):
    if request.method == "GET":

        if request.user.pladatuser.is_student():
            return HttpResponseForbidden("Invalid user")  # TODO

        job = get_object_or_404(Job, pk=id)
        recruiter = request.user.pladatuser.recruiter
        if recruiter != job.recruiter:
            return HttpResponseForbidden("Invalid user")  # TODO

        applications = AppliedJob.objects.filter(
            job=job,
            student_status=Response.INTERESTED,
            recruiter_status=Response.INTERESTED,
        ).order_by("-match_rate")

        ctx = {
            "applications": applications,
            "job": job,
        }

        return render(request, "job_matches.html", context=ctx)

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
    if recruiter != job.recruiter:
        return HttpResponseForbidden("Invalid user") # TODO "Where"
    ctx = {
        "job": job,
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
    if request.user.pladatuser.is_student():
        # TODO: Return some other page
        return HttpResponseForbidden("Invalid user")

    job = get_object_or_404(Job, id=id)
    recruiter = request.user.pladatuser.recruiter
    if recruiter != job.recruiter:
        return HttpResponseForbidden("Invalid User")  # TODO

    ctx = {
        "job": job,
        "is_student": False,
    }

    if request.method == "GET":
        ctx["is_owner"] = request.user.pladatuser.recruiter == job.recruiter
        return render(request, "job.html", context=ctx)
    else:
        return HttpResponseForbidden("Forbidden method")
