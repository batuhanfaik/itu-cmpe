# Generated by Django 3.1.5 on 2021-01-16 17:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('student', '0001_initial'),
        ('recruiter', '0001_initial'),
        ('job', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='recruiter',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recruiter.recruiter'),
        ),
        migrations.AddField(
            model_name='appliedjob',
            name='applicant',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='student.student'),
        ),
        migrations.AddField(
            model_name='appliedjob',
            name='job',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='job.job'),
        ),
        migrations.AlterUniqueTogether(
            name='appliedjob',
            unique_together={('applicant', 'job')},
        ),
    ]
