from django.shortcuts import render

from apps.pladat.models import PladatUser
from apps.pladat.models import User
# Create your views here.



def applicant_profile(request):
    user = User.objects.get(email='test@pladat.com')
    applicant = PladatUser.objects.get(user=user)
    ctx = {'applicant':applicant} #This should contain the PladatUser of the student
    if request.method == 'GET':
        print(ctx)
        return render(request, 'applicant_profile.html', context=ctx)
    if request.method == 'POST':
        if 'yes' in request.POST: # interested button is clicked
            print('yes')
        elif 'no' in request.POST: # not interested button is clicked
            print('no')
        return render(request, 'applicant_profile.html', context=ctx)