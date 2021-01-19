import numpy as np
import lightgbm as lgb
import string
import pickle
from apps.student.models import Student
from apps.job.models import AppliedJob, Job

embeddings = None

def doc2vec(inp, embeddings):
    """
    Inputs:
    @inp: Input string to be converted into vector
    @embeddings: Dictionary keeping all the embeddings for vocabulary
    
    Outputs:
    Returns normalized embedding vector for the given inp
    """
    print("a8")
    inp = inp.translate(str.maketrans('', '', string.punctuation))
    inp = inp.lower()
    inp_val = np.zeros((50,), dtype=np.float64)
    inp_len = len(inp.split())
    print("a9")
    for w in inp.split():
        inp_val += embeddings.get(w, inp_val)
    print("a10")
    retval = inp_val / inp_len
    print("a11")
    return np.reshape(retval, (1, 50))

def create_features(student, job):
    """
    Inputs: student and job object from the database
    Outputs: Feature vector that can work with the ML model
    """
    global embeddings
    if embeddings is None:
        print("a1")
        with open("apps/recommend/embeddings.p", "rb") as f:
            embeddings = pickle.load(f)
        print("a2")

    features = np.zeros((1  , 113)) # Total feature length 113

    skills, degree, major, university = student.skills.all(), student.degree, student.major, student.university
    years_worked = student.years_worked
    # Processing skills
    skills = student.skills.all()
    skill_names = ["Machine Learning", "Computer Vision", "Computer Security", "Software Engineering",
    "Algorithms", "Statistics", "Web Development", "Systems Programming", "Computer Communications"]
    skill_len = len(skill_names)
    for i in range(skill_len):
        if len(skills.filter(name=skill_names[i])) == 1:
            features[0, i] = 1

    print("a3")
    print("a4-1")
    # Other fields are easier
    if degree == "msc":
        features[0, skill_len] = 1
    print("a4-2")
    if major == "math":
        features[0, skill_len + 1] = 1
    print("a4-3")
    if university == "boun":
        features[0, skill_len + 2] = 1
    print("a4-4")
    if university == "koc":
        features[0, skill_len + 2] = 2
    
    print("a4-5")

    features[0, skill_len + 3] = years_worked
    skill_len += 4

    # Processing jobs
    print("a5")
    features[0, skill_len:skill_len+50] = doc2vec(job.title, embeddings)
    print(features[0, (skill_len+50):].shape)
    features[0, (skill_len+50):] = doc2vec(job.description, embeddings)
    print("a6")

    return features
