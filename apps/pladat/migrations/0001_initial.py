# Generated by Django 3.1.5 on 2021-01-09 17:19

from django.db import migrations, models
import django.db.models.deletion
import django_countries.fields
import phonenumber_field.modelfields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='PladatUser',
            fields=[
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='auth.user', verbose_name='User')),
                ('first_name', models.CharField(max_length=128, verbose_name='First name')),
                ('last_name', models.CharField(max_length=128, verbose_name='Last name')),
                ('phone_number', phonenumber_field.modelfields.PhoneNumberField(max_length=128, region=None, verbose_name='Phone number')),
                ('address', models.CharField(max_length=128, verbose_name='Address')),
                ('city', models.CharField(max_length=128, verbose_name='City')),
                ('state', models.CharField(max_length=128, null=True, verbose_name='State')),
                ('country', django_countries.fields.CountryField(max_length=2, verbose_name='Country')),
                ('user_type', models.IntegerField(choices=[(0, 'Student account'), (1, 'Company account')], verbose_name='User type')),
            ],
        ),
    ]
