import numpy as np
import pandas as pd
import lightgbm as lgb
import string
import pickle
from apps.student.models import Student
from apps.job.models import AppliedJob, Job

def doc2vec(inp, embeddings):
    """
    Inputs:
    @inp: Input string to be converted into vector
    @embeddings: Dictionary keeping all the embeddings for vocabulary
    
    Outputs:
    Returns normalized embedding vector for the given inp
    """

    inp = inp.translate(str.maketrans('', '', string.punctuation))
    inp = inp.lower()
    inp_val = np.zeros((50,), dtype=np.float64)
    inp_len = len(inp.split())
    for w in inp.split():
        inp_val += embeddings.get(w, inp_val)
    return inp_val / inp_len

def create_features(student, job):
    """
    Inputs: student and job object from the database
    Outputs: Feature vector that can work with the ML model
    """
    embeddings = None
    with open("embeddings.p", "rb") as f:
        embeddings = pickle.load(embeddings)

    features = np.zeros((1, 113)) # Total feature length 113

    # Processing skills
    skills = student.skills.all()
    skill_names = ["Machine Learning", "Computer Vision", "Computer Security", "Software Engineering",
    "Algorithms", "Statistics", "Web Development", "Systems Programming", "Computer Communications"]
    skill_len = len(skill_names)
    for i in range(skill_len):
        if len(skills.filter(name=skill_names[i])) == 1:
            features[0, i] = 1

    # Other fields are easier
    if student.degree.get_degree_display == "Master of Science":
        features[skill_len + 1] = 1
    if student.major.get_major_display == "Mathematics":
        features[skill_len + 2] = 1
    if student.university.get_university_display == "Bogazici University":
        features[skill_len + 3] = 1
    elif student.university.get_university_display == "Koc University":
        features[skill_len + 3] = 2

    features[skill_len + 4] = student.years_worked

    skill_len += 5

    # Processing jobs
    features[i, skill_len:skill_len+50] = doc2vec(job.title, embeddings)
    features[i, skill_len+50:] = doc2vec(job.description, embeddings)

    return features
