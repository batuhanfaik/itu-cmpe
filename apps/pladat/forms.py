
from .models import PladatUser
from django import forms


class RegistrationForm(forms.ModelForm):
    STUDENT = 0
    COMPANY = 1
    USER_TYPE = (
        (STUDENT, 'I want to find internships'),
        (COMPANY, 'I want to find students for my job posting'),
    )
    user_type = forms.ChoiceField(choices=USER_TYPE)
    email = forms.EmailField(label='Please enter your email', required=True)
    password = forms.CharField(label='Please enter your password', required=True)

    class Meta:
        model = PladatUser
        fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country']