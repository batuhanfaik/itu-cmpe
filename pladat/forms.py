from django import forms

class RegistrationForm(forms.Form):
    email = forms.EmailField(label='Please enter your email', required=True)
    password = forms.CharField(widget=forms.PasswordInput(), label='Please enter your password', required=True)