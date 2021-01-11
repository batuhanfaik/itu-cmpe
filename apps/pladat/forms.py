
from .models import PladatUser
from django import forms

class RegistrationForm(forms.ModelForm):
    email = forms.EmailField(label='Please enter your email', required=True, 
        help_text="E-mail"
        )
    password = forms.CharField(label='Please enter your password', required=True,
        help_text="Password"
    )

    class Meta:
        model = PladatUser
        fields = ['first_name', 'last_name', 'phone_number', 'address', 'city', 'state', 'country', 'user_type']
            
    def __init__(self, *args, **kwargs):
        super(RegistrationForm, self).__init__(*args, **kwargs)
        for k,v in self.fields.items():
            # HTML attributes to the form fields can be added here
            v.widget.attrs['class'] = 'form-control'
            v.widget.attrs['placeholder'] = v.help_text
            if k=='country':
                v.widget.empty_label = "Select a country"

class LoginForm(forms.Form):
    email = forms.EmailField(label='Please enter your email', required=True, 
        help_text="E-mail"
    )
    password = forms.CharField(label='Please enter your password', required=True,
        help_text="Password"
    )

