from django import forms

from apps.pladat.models import PladatUser

from .models import Student

class UpdateStudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = '__all__'
        exclude = ('pladatuser',)

    def __init__(self, *args, **kwargs):
        super(UpdateStudentForm, self).__init__(*args, **kwargs)
        for k, v in self.fields.items():
            # HTML attributes to the form fields can be added here
            v.widget.attrs['class'] = 'form-control'
            if k == 'skills':
                v.widget.attrs['class'] = 'form-control selectpicker'
                v.widget.attrs['title'] = 'Select Skills'
