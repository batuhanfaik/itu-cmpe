from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render

from django.contrib.auth import authenticate, login

def main_page_view(request):
    ctx = {}

    if not request.user.is_authenticated:
        # Check if user logged in
        ctx = {'user': 'guest'}

    return render(request, "main_page.html", context=ctx)