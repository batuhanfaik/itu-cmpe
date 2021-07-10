import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os

VAL_SIZE = 20


def write_to_file(train_p, val_p, temp_df, train, val):
    if "splits" not in os.listdir("."):
        os.mkdir("splits")
    f_name = "splits/split_{}-{}.csv".format(train_p, val_p)

    for i in train:
        temp_df.loc[i, "Dataset_type"] = "TRAIN"

    for i in val:
        temp_df.loc[i, "Dataset_type"] = "VAL"

    temp_df.to_csv(f_name, index=False)


df_original = pd.read_csv("Chest_xray_Corona_Metadata.csv")
df = pd.read_csv("without_test.csv")

image_id, label = df["X_ray_image_name"], df["Label"]

val_p = VAL_SIZE/100
train_p = 1 - val_p

train_set, val_set = train_test_split(image_id, test_size=val_p, random_state=1773)

temp_df = df_original.copy()
write_to_file(train_p, val_p, temp_df, train_set.index, val_set.index)
