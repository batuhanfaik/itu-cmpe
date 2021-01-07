
from .models import PladatUser
from django import forms


class RegistrationForm(forms.ModelForm):
    email = forms.EmailField(label='Please enter your email', required=True)
    password = forms.CharField(label='Please enter your password', required=True)

    class Meta:
        model = PladatUser
        fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country', 'user_type']