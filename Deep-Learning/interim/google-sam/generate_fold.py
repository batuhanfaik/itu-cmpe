import numpy as np
import sys
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def write_to_file(fold, train, val):
    if "folds" not in os.listdir("."):
        os.mkdir("folds")
    fname_train = "folds/fold_" + str(fold) + "_train.txt"
    fname_val = "folds/fold_" + str(fold) + "_val.txt"

    with open(fname_train, 'w') as file:
        for t in train:
            data = t + "\n"
            file.write(data)

    with open(fname_val, 'w') as file:
        for t in val:
            data = t + "\n"
            file.write(data)


DATASET_PATH = "/mnt/sdb1/datasets/cassava-leaf-disease-classification"
df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))

# Train: 90%, Test: 10%
image_id, label = df["image_id"], df["label"]
train, test, train_label, test_label = train_test_split(image_id, label, test_size=0.1,
                                                        random_state=1773)

# Write the test file
with open("test.txt", 'w') as file:
    for t in test:
        data = t + "\n"
        file.write(data)

kf = KFold(n_splits=5, shuffle=True, random_state=2020)
kf.get_n_splits(train)

fold = 1
for train_index, val_index in kf.split(train):
    image_train, image_val = train.iloc[train_index], train.iloc[val_index]
    label_train, label_val = train_label.iloc[train_index], train_label.iloc[val_index]
    write_to_file(fold, image_train, image_val)
    fold += 1
