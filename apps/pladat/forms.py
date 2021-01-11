
from .models import PladatUser
from django import forms

# def create_widget(fields):
#     widgets = {}
#     for field in fields:
#         if field is not "user_type":
#             widgets[field] = forms.TextInput(attrs={'class' : 'form-control'})
#         else:
#             widgets[field] = forms.Select(attrs={'class' : 'form-control'})
#     return widgets


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
        # for visible in self.visible_fields():
        for k,v in self.fields.items():
            # v.widget.attrs['placeholder'] = k.capitalize()
            v.widget.attrs['class'] = 'form-control'
            v.widget.attrs['placeholder'] = v.help_text
            if k=='country':
                v.widget.empty_label="TEST"