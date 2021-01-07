from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render

import json

def main_page_view(request):
    return render(request, "main_page.html", context={})