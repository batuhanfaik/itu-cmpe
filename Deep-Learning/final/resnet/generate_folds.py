import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import os

def write_to_file(fold, temp_df, train, val):
    if "folds" not in os.listdir("."):
        os.mkdir("folds")
    fname = "folds/fold_" + str(fold) + ".csv"


    for i in train:
        temp_df.loc[i,'Dataset_type'] = "TRAIN"

    for i in val:
        temp_df.loc[i,'Dataset_type'] = "VAL"
    
    temp_df.to_csv(fname,index=False)


df_original = pd.read_csv("Chest_xray_Corona_Metadata.csv")
df = pd.read_csv("without_test.csv")

image_id, label = df["X_ray_image_name"], df["Label"]

kf = KFold(n_splits=5, shuffle=True, random_state=2020)
kf.get_n_splits(image_id)

fold = 1
for train_index, val_index in kf.split(image_id):
    image_train, image_val = image_id.iloc[train_index], image_id.iloc[val_index]
    temp_df = df_original.copy()
    tr_index = image_train.index
    val_index = image_val.index
    write_to_file(fold, temp_df, tr_index, val_index)
    fold += 1