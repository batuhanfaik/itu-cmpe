
from .models import Student
from apps.pladat.models import PladatUser
from django import forms

class UpdatePladatUserForm(forms.ModelForm):
    class Meta:
        model = PladatUser
        fields = '__all__'
        exclude = ('user', 'user_type')

class UpdateStudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = '__all__'
        exclude = ('pladatuser',)

