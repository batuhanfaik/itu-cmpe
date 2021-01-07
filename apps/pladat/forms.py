from django import forms

USER_TYPE = (
    (0, 'I want to find internships'),
    (1, 'I want to find students for my job posting'),
)

class RegistrationForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}), label='Please enter your email', required=True)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='Please enter your password', required=True)
    user_type = forms.ChoiceField(widget=forms.Select(attrs={'class': 'form-control'}), choices=USER_TYPE)