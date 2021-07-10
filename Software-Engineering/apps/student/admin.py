from django.contrib import admin

from .models import Skill, Student

# Register your models here.

admin.site.register(Student)
admin.site.register(Skill)
