# Generated by Django 3.1.5 on 2021-01-16 15:24

from django.db import migrations, models
import django.db.models.deletion
import django_countries.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('recruiter', '0001_initial'),
        ('student', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Job',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(help_text='Title', max_length=128)),
                ('description', models.TextField(help_text='Description', max_length=512)),
                ('requirements', models.TextField(help_text='Requirements', max_length=512)),
                ('city', models.CharField(help_text='City', max_length=128)),
                ('state', models.CharField(help_text='State', max_length=128, null=True)),
                ('country', django_countries.fields.CountryField(help_text='Country', max_length=2)),
                ('recruiter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recruiter.recruiter')),
            ],
        ),
        migrations.CreateModel(
            name='AppliedJob',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('applicant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='student.student')),
                ('job', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='job.job')),
            ],
        ),
    ]
