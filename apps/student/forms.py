from .models import Student
from apps.pladat.models import PladatUser
from django import forms


class UpdatePladatUserForm(forms.ModelForm):
    class Meta:
        model = PladatUser
        fields = '__all__'
        exclude = ('user', 'user_type')

    def __init__(self, *args, **kwargs):
        super(UpdatePladatUserForm, self).__init__(*args, **kwargs)
        for k, v in self.fields.items():
            # HTML attributes to the form fields can be added here
            v.widget.attrs['class'] = 'form-control'


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
