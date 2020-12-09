import numpy as np
import sys
import random
import pandas as pd
from sklearn.model_selection import KFold

random.seed(324234)

def write_to_file(fold, train, test):
    fname_train = "fold_" + str(fold) + "_train.txt"
    fname_test = "fold_" + str(fold) + "_test.txt"

    with open(fname_train, 'w') as file:
        for t in train:
            data = t + "\n"
            file.write(data)
            
    
    with open(fname_test, 'w') as file:
        for t in test:
            data = t + "\n"
            file.write(data)
            



df = pd.read_csv("train.csv")
images = df['image_id']



kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(images)

fold = 1

for train_index, test_index in kf.split(images):
    
    X_train, X_test = images[train_index], images[test_index]
    write_to_file(fold, X_train, X_test)
    fold += 1