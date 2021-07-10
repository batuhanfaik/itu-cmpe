import shutil
import numpy as np
import torch
import time
import re
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loader import DataReader
import torch.nn as nn
import os
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report
from densenet import densenet121

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

def conf_matrix(cmat):
    arr = cmat.copy()
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(arr.shape[0]):
        tp += arr[i, i]
        tn += arr.sum() - arr[:, i].sum() - arr[i, :].sum() + arr[i, i]
        fn += arr[i, :].sum() - arr[i, i]
        fp += arr[:, i].sum() - arr[i, i]

    return tn, fp, fn, tp


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
num_workers = 1

'''
multi class classificaiton:
    multi_to_multi = True
    multi_class = True
multi  to binary classificaiton:
    multi_to_multi = False
    multi_class = True
binary classificaiton:
    multi_to_multi = False
    multi_class = False
'''
#####################################################
# CRX Normalization parameters
#####################################################
crx_norm = {
    "clip_limit": 3.0,
    "tile_grid_size": (10, 10),
    "median_filter_size": 5,
    "percentiles": (2, 98)
}
#####################################################

#####################################################
multi_to_multi = True
multi_class = True

oversample = True
#####################################################
dataset_path = "/mnt/sdb1/datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
split_path = "splits/split_0.8-0.2.csv"


test_loader = torch.utils.data.DataLoader(
    DataReader(mode='test', path=split_path, dataset_path=None, oversample=oversample,
               multi_class=multi_class, crx_norm=None),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers)

num_classes = 5
num_epochs = 50

model_path = "multi_class_classification_baseline/models/"

model_list = os.listdir(model_path)
model_list = sorted(model_list)

#model_list = ["model_epoch_7.pt"]

for mname in model_list:
    print("model name : ", mname)
    pth = model_path + mname
    model = torch.load(pth)
    

    model = model.to(device)

    if multi_class == True:
        loss = torch.nn.CrossEntropyLoss().to(device)
        loss_test = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        loss = torch.nn.BCEWithLogitsLoss().to(device)    
    with torch.no_grad():

        model.eval()
        total_tn = 0
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_true = 0.0
        total_false = 0.0
        y_true = None
        y_pred = None
        time_start = time.time()

        for i, batch in enumerate(test_loader):
            img = batch['image']
            img_class = batch['label']
            img = img.float()
            img = img.to(device)
            if multi_class == False:
                img_class = img_class.float()
            #img.requires_grad = False
            img_class = img_class.to(device)
            #img_class.requires_grad = False
            output = model(img)

            if multi_to_multi == True:
                prediction = torch.argmax(output.data, 1)
                
            elif multi_class == True:  # multi to binary
                temp_list = []
                temp_list = torch.cat([output[:, 0:1].sum(dim=1, keepdim=True),
                                       output[:, 1:3].sum(dim=1, keepdim=True)], dim=1)
                new_output = temp_list

                res = torch.sigmoid(new_output)

                img_class[img_class > 0] = 1
                #img_class = img_class.int()

                prediction = torch.argmax(new_output.data, 1)
                
            else:  # binary
                res = torch.sigmoid(output)
                img_class = img_class.int()
                temp_output = res
                temp_output = (temp_output > 0.5).int()
                prediction = temp_output

            if y_pred is None:
                y_pred = prediction.cpu().numpy().flatten()
            else:
                y_pred = np.concatenate([y_pred, prediction.cpu().numpy().flatten()])
            
            if y_true is None:
                y_true = img_class.cpu().numpy().flatten()
            else:
                y_true = np.concatenate([y_true, img_class.cpu().numpy().flatten()])                    

        print("Test scores:")
        if multi_to_multi == True:    
            report = classification_report(y_true, y_pred, labels = [0,1,2], output_dict=False)
        else:
            report = classification_report(y_true, y_pred, labels = [0,1], output_dict=False)
        print(report)

