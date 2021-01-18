from django import forms

from .models import PladatUser


class RegistrationForm(forms.ModelForm):
    email = forms.EmailField(label='Please enter your email', required=True,
                             help_text="E-mail", widget=forms.EmailInput
                             )
    password = forms.CharField(label='Please enter your password', required=True,
                               help_text="Password", widget=forms.PasswordInput
                               )

    class Meta:
        model = PladatUser
        fields = '__all__'
        exclude = ('pladatuser', 'user')

    def __init__(self, *args, **kwargs):
        super(RegistrationForm, self).__init__(*args, **kwargs)
        for k, v in self.fields.items():
            # HTML attributes to the form fields can be added here
            v.widget.attrs['class'] = 'form-control'
            v.widget.attrs['placeholder'] = v.help_text
            if k == 'country' or k == 'user_type':
                v.widget.attrs['style'] = "height: auto !important;"


class LoginForm(forms.Form):
    email = forms.EmailField(label='Please enter your email', required=True,
                             help_text="E-mail", widget=forms.EmailInput
                             )

    password = forms.CharField(label='Please enter your password', required=True,
                               help_text="Password", widget=forms.PasswordInput
                               )

    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)
        for k, v in self.fields.items():
            # HTML attributes to the form fields can be added here
            v.widget.attrs['class'] = 'form-control'
            v.widget.attrs['placeholder'] = v.help_text
