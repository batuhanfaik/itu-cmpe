# Generated by Django 3.1.5 on 2021-01-16 13:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('recruiter', '0001_initial'),
        ('job', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='recruiter',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recruiter.recruiter'),
        ),
    ]
