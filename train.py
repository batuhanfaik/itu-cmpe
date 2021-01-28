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
import resnet
from preprocessor import Preprocessor


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


def save_res(epoch_id, total_loss, loader_len, report, time_start, res_name, mode, multi_class, multi_to_multi):
    with open(res_name, "a") as f:
        f.write(mode)
        f.write(": ")
        f.write(str(datetime.datetime.now()))
        f.write("\n")

        f.write("Epoch ")
        f.write(" scores: ")
        f.write(str(epoch_id))
        f.write("\n")

        f.write("Loss: ")
        f.write(str((total_loss / loader_len)))
        f.write("\n")
        f.write("\n")
        f.write(report)
        f.write("\n")
        f.write("Time (s): ")
        f.write(str(time.time() - time_start))
        f.write("\n")
        f.write("--------------------------------------")
        f.write("\n")


def prepare_experiment(project_path=".", experiment_name=None):
    next_experiment_number = 0
    for directory in os.listdir(project_path):
        search_result = re.search("experiment_(.*)", directory)
        if search_result and next_experiment_number < int(search_result[1]) + 1:
            next_experiment_number = int(search_result[1]) + 1

    if not experiment_name:
        exp_name = "experiment_{}".format(next_experiment_number)
    else:
        exp_name = experiment_name
        if os.path.exists(experiment_name) and os.path.isdir(experiment_name):
            shutil.rmtree(exp_name)

    os.mkdir(exp_name)
    os.mkdir(exp_name + '/graphs/')
    os.mkdir(exp_name + '/models/')
    os.mkdir(exp_name + '/code/')

    return exp_name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
num_workers = 1

'''
multi class classification:
    multi_to_multi = True
    multi_class = True
multi to binary classification:
    multi_to_multi = False
    multi_class = True
binary classification:
    multi_to_multi = False
    multi_class = False
'''
dataset_path = "/mnt/sdb1/datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
preprocessed_dataset_path = "/mnt/sdb1/datasets/Coronahack-Chest-XRay-Dataset/preprocessed_dataset"
split_path = "splits/split_0.8-0.2.csv"
#####################################################
# CRX Normalization parameters
#####################################################
crx_norm = {
    "clip_limit": 2.0,
    "tile_grid_size": (8, 8),
    "median_filter_size": 5,
    "percentiles": (2, 98)
}
#####################################################
# Dataset Preprocessing
#####################################################
# preprocessor = Preprocessor(dataset_path=dataset_path, crx_params=crx_norm, mode="c")
# dataset_path, dataset_mean, dataset_std = preprocessor.preprocess_dataset()
#####################################################
# Default mean and std values
#####################################################
# dataset_mean = 123
# dataset_std = 57
#####################################################
# Preprocessed mean and std values
#####################################################
dataset_mean = 143
dataset_std = 72
dataset_path = preprocessed_dataset_path
#####################################################
multi_to_multi = False
multi_class = False

oversample = True
#####################################################

train_loader = torch.utils.data.DataLoader(
    DataReader(mode='train', path=split_path, dataset_path=dataset_path, oversample=oversample,
               multi_class=multi_class, mean=dataset_mean, std=dataset_std, crx_norm=None),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(
    DataReader(mode='val', path=split_path, dataset_path=dataset_path, oversample=oversample,
               multi_class=multi_class, mean=dataset_mean, std=dataset_std, crx_norm=None),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

experiment_name = prepare_experiment(experiment_name="norm_binary_crx")
res_name = experiment_name + "/" + experiment_name + "_res.txt"

all_python_files = os.listdir('.')

for i in range(len(all_python_files)):
    if '.py' in all_python_files[i]:
        os.system('cp ' + all_python_files[i] + ' ' + experiment_name + '/code/')

num_classes = 5
num_epochs = 50

model = resnet.resnet101(pretrained=False)
num_features_resnet = model.in_features

if multi_class == True:
    model.fc = nn.Linear(num_features_resnet, 3)
else:
    model.fc = nn.Linear(num_features_resnet, 1)

model = model.to(device)

lr = 1e-3
# base_optimizer = torch.optim.SGD
base_optimizer = torch.optim.Adam
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-10)

if multi_class == True:
    loss = torch.nn.CrossEntropyLoss().to(device)
    loss_test = torch.nn.BCEWithLogitsLoss().to(device)
else:
    loss = torch.nn.BCEWithLogitsLoss().to(device)


all_tr_losses = torch.zeros(num_epochs, 1)
all_test_losses = torch.zeros(num_epochs, 1)


for epoch_id in range(1, num_epochs + 1):
    model.train()

    total_loss = 0
    y_true = None
    y_pred = None
    time_start = time.time()

    for i, data in enumerate(train_loader):
        img = data['image']
        img_class = data['label']
        img = img.float()
        if multi_class == False:
            img_class = img_class.float()

        img = img.to(device)
        img.requires_grad = True

        img_class = img_class.to(device)
        img_class.requires_grad = False
        optimizer.zero_grad()

        output = model(img)
        if multi_class == True:
            loss_value = loss(output, img_class)
            prediction = torch.argmax(output.data, 1)
            
        else:
            loss_value = loss(output, img_class.unsqueeze(1))#if multiclass false
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

        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.data
        
        if (i + 1) % 100 == 0:
            print("Pre-report Epoch:", epoch_id)
            print("Loss: %f" % loss_value.data)
            print("Status -> %d / %d" % (i + 1, len(train_loader)))
            print("************************************")
    
    if multi_class == True:
        report = classification_report(y_true, y_pred, labels = [0,1,2], output_dict=True)
        report_to_print = classification_report(y_true, y_pred, labels = [0,1,2], output_dict=False)
    else:
        report = classification_report(y_true, y_pred, labels = [0,1], output_dict=False)
        report_to_print = classification_report(y_true, y_pred, labels = [0,1], output_dict=False)
    
    print("Train:", datetime.datetime.now())
    print("Epoch %d scores:" % epoch_id)
    print("Loss: %f" % (total_loss / len(train_loader))) 
    print("Report: \n", report_to_print)
    print("LR: %f" % get_lr(optimizer))
    print("Time (s): " + str(time.time() - time_start))
    print("--------------------------------------")

    # all_tr_losses[epoch_id] = total_loss.cpu()
    all_tr_losses[epoch_id-1] = total_loss / len(train_loader)


    save_res(epoch_id, total_loss, len(train_loader), report_to_print, time_start, res_name, "train", multi_class, multi_to_multi)
    
    with torch.no_grad():

        model.eval()

        val_losses = 0
        total_loss = 0
        y_true = None
        y_pred = None
        time_start = time.time()

        for i, batch in enumerate(val_loader):
            img = batch['image']
            img_class = batch['label']
            img = img.float()
            img = img.to(device)
            if multi_class == False:
                img_class = img_class.float()
            img_class = img_class.to(device)
            output = model(img)

            if multi_to_multi == True:
                prediction = torch.argmax(output.data, 1)
                val_loss = loss(output, img_class)
                
            elif multi_class == True:  # multi to binary
                

                temp_list = []
                temp_list = torch.cat([output[:, 0:1].sum(dim=1, keepdim=True),
                                       output[:, 1:3].sum(dim=1, keepdim=True)], dim=1)
                new_output = temp_list
                val_loss = loss(output, img_class)
                res = torch.sigmoid(new_output)

                #img_class = img_class.detach().cpu().numpy()
                #img_class = img_class.flatten().astype(np.int)
                img_class[img_class > 0] = 1
                

                prediction = torch.argmax(new_output.data, 1)
                
                    
            else:  # binary
                val_loss = loss(output, img_class.unsqueeze(1))
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


            val_losses += val_loss
            total_loss += val_loss.data

        if multi_to_multi == True:
            report = classification_report(y_true, y_pred, labels = [0,1,2], output_dict=True)
            report_to_print = classification_report(y_true, y_pred, labels = [0,1,2], output_dict=False)
        else:
            report = classification_report(y_true, y_pred, labels = [0,1], output_dict=True)
            report_to_print = classification_report(y_true, y_pred, labels = [0,1], output_dict=False)


        model_path = experiment_name + "/models/model_epoch_" + str(epoch_id) + '.pt'
        torch.save(model, model_path)            

        print("Test:", datetime.datetime.now())
        print("Val %d scores:" % epoch_id)
        print("Loss %f" % (total_loss / len(val_loader)))
        print("Report:\n ", report_to_print)
        print("Time (s): " + str(time.time() - time_start))
        print("--------------------------------------")

        # all_test_losses[epoch_id] = total_loss.cpu()
        all_test_losses[epoch_id-1] = total_loss / len(val_loader)

    scheduler.step(val_losses / len(val_loader))

    save_res(epoch_id, total_loss, len(train_loader), report_to_print, time_start, res_name, "train", multi_class, multi_to_multi)

    training_loss = all_tr_losses.numpy()
    training_loss = np.reshape(training_loss, (training_loss.shape[1] * training_loss.shape[0], -1))

    val_loss = all_test_losses.numpy()
    val_loss = np.reshape(val_loss, (val_loss.shape[1] * val_loss.shape[0], -1))

    training_loss[training_loss == 0] = np.nan
    val_loss[val_loss == 0] = np.nan

    plt.plot(training_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title("Train - Validation Loss Curve")
    plt.ylabel('Loss')
    plt.xlabel('# Epochs')
    plt.legend()
    plt.grid(True)
    fig_path = experiment_name + "/graphs/train_val_loss_epoch_" + str(epoch_id) + '.png'
    plt.savefig(fig_path)
    plt.clf()