import numpy as np
import pandas as pd
import lightgbm as lgb
import string
import pickle

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

def create_features(user, job):
    pass



