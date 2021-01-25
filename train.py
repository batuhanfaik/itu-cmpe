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
from sklearn.metrics import confusion_matrix
from densenet import densenet121


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


def save_res(epoch_id, total_loss, loader_len, acc, sens, spec, time_start, res_name, mode):
    with open(res_name, "a") as f:
        f.write(mode)
        f.write(": ")
        f.write(str(datetime.datetime.now()))
        f.write("\n")

        f.write("Epoch ")
        # f.write(str(i))
        f.write(" scores: ")
        f.write(str(epoch_id))
        f.write("\n")

        f.write("Loss: ")
        f.write(str((total_loss / loader_len)))
        f.write("\n")

        f.write("Acc: ")
        f.write(str(acc))
        f.write("\n")

        f.write("Sensitivity: ")
        f.write(str(sens))
        f.write("\n")    

        f.write("Specificity: ")
        f.write(str(spec))
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
multi_to_multi = False
multi_class = True

oversample = False
#####################################################
dataset_path = "/mnt/sdb1/datasets/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
split_path = "splits/split_0.8-0.2.csv"

train_loader = torch.utils.data.DataLoader(
    DataReader(mode='train', path=split_path, dataset_path=None, oversample=oversample,
               multi_class=multi_class, crx_norm=None), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers)

val_loader = torch.utils.data.DataLoader(
    DataReader(mode='val', path=split_path, dataset_path=None, oversample=oversample,
               multi_class=multi_class, crx_norm=None),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers)

experiment_name = prepare_experiment()
res_name = experiment_name + "/" + experiment_name + "_res.txt"

all_python_files = os.listdir('.')

for i in range(len(all_python_files)):
    if '.py' in all_python_files[i]:
        os.system('cp ' + all_python_files[i] + ' ' + experiment_name + '/code/')

num_classes = 5
num_epochs = 100

model = densenet121(pretrained=False)
num_features_dense = model.classifier.in_features

if multi_class == True:
    model.classifier = nn.Linear(num_features_dense, 4)
else:
    model.classifier = nn.Linear(num_features_dense, 1)

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
all_tr_accuracies = np.zeros((num_epochs, 1))
all_test_losses = torch.zeros(num_epochs, 1)
all_test_accuracies = np.zeros((num_epochs, 1))

for epoch_id in range(1, num_epochs + 1):
    model.train()

    total_loss = 0
    total_true = 0
    total_false = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    time_start = time.time()

    for i, data in enumerate(train_loader):
        img = data['image']
        img_class = data['label']
        img = torch.Tensor(img.float())
        if multi_class == False:
            img_class = torch.Tensor(img_class.float())

        img = img.to(device)
        img.requires_grad = True

        img_class = img_class.to(device)
        img_class.requires_grad = False
        optimizer.zero_grad()

        output = model(img)
        if multi_class == True:
            _, prediction = torch.max(output.data, 1)
            loss_value = loss(output, img_class)
            arr = confusion_matrix(img_class.cpu().numpy(), prediction.cpu().numpy(), labels=[0, 1, 2, 3])
            tn, fp, fn, tp = conf_matrix(arr)

            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp            
        else:
            temp_output = output.clone()
            temp_output[temp_output > 0.5] = 1
            temp_output[temp_output <= 0.5] = 0
            prediction = temp_output.clone()
            loss_value = loss(output, img_class.unsqueeze(1))  # if multiclass false
            tn, fp, fn, tp = confusion_matrix(img_class.cpu().numpy(), prediction.cpu().numpy()).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp

        # loss_value = loss(output, img_class.unsqueeze(1))#if multiclass false
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.data
        #total_true += torch.sum(prediction == img_class.data)
        #total_false += torch.sum(prediction != img_class.data)

        if (i + 1) % 100 == 0:
            print("Pre-report Epoch:", epoch_id)
            print("Loss: %f" % loss_value.data)
            print("Status -> %d / %d" % (i + 1, len(train_loader)))
            print("************************************")

    #acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
    temp_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    sens = (total_tp / (total_tp + total_fn))
    spec = (total_tn / (total_tn + total_fp))
    print("Train:", datetime.datetime.now())
    print("Epoch %d scores:" % epoch_id)
    print("Loss: %f" % (total_loss / len(train_loader)))
    #print("Accuracy: %f" % acc)
    print("Accuracy %f" % temp_acc)
    print("Sensitivity %f" % sens)
    print("Specificity %f" % spec)    
    print("LR: %f" % get_lr(optimizer))
    print("Time (s): " + str(time.time() - time_start))
    print("--------------------------------------")

    # all_tr_losses[epoch_id] = total_loss.cpu()
    all_tr_losses[epoch_id] = total_loss / len(train_loader)
    all_tr_accuracies[epoch_id] = temp_acc

    save_res(epoch_id, total_loss, len(train_loader), temp_acc, sens, spec, time_start, res_name, "train")

    with torch.no_grad():

        model.eval()

        val_losses = 0
        total_loss = 0
        total_true = 0
        total_false = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        total_tp = 0
        mtm_conf = []
        time_start = time.time()

        for i, batch in enumerate(val_loader):
            img = batch['image']
            img_class = batch['label']
            img = torch.Tensor(img.float())
            img = img.to(device)
            if multi_class == False:
                img_class = torch.Tensor(img_class.float())
            img.requires_grad = False
            # img_class = torch.Tensor(img_class.float())
            img_class = img_class.to(device)
            img_class.requires_grad = False
            output = model(img)

            if multi_to_multi == True:
                _, prediction = torch.max(output.data, 1)
                val_loss = loss(output, img_class)
                arr = confusion_matrix(img_class.cpu().numpy(), prediction.cpu().numpy(), labels=[0, 1, 2, 3])
                tn, fp, fn, tp = conf_matrix(arr)

                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp
            elif multi_class == True:  # multi to binary

                temp_list = []
                temp_list = torch.cat([output[:, 0:1].sum(dim=1, keepdim=True),
                                       output[:, 1:4].sum(dim=1, keepdim=True)], dim=1)
                new_output = temp_list

                img_class[img_class > 0] = 1

                # new_output[new_output > 0.5] = 1
                # new_output[new_output <= 0.5] = 0
                # prediction = new_output.clone()
                _, prediction = torch.max(new_output.data, 1)
                val_loss = loss(new_output, img_class)

                tn, fp, fn, tp = confusion_matrix(img_class.cpu().numpy(), prediction.cpu().numpy()).ravel()
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp
            else:  # binary

                temp_output = output.clone()
                temp_output[temp_output > 0.5] = 1
                temp_output[temp_output <= 0.5] = 0
                prediction = temp_output.clone()
                val_loss = loss(output, img_class.unsqueeze(1))  # if multiclass false
                tn, fp, fn, tp = confusion_matrix(img_class.cpu().numpy(), prediction.cpu().numpy()).ravel()
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp

            # val_loss = loss(output, img_class))#if multiclass false
            val_losses += val_loss
            total_loss += val_loss.data
            #total_true += torch.sum(prediction == img_class.data)
            #total_false += torch.sum(prediction != img_class.data)

        #acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
        temp_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        sens = (total_tp / (total_tp + total_fn))
        spec = (total_tn / (total_tn + total_fp))
        if temp_acc > 0.65:
            model_path = experiment_name + "/models/model_epoch_" + str(epoch_id) + '.pt'
            torch.save(model, model_path)
        print("Test:", datetime.datetime.now())
        print("Val %d scores:" % epoch_id)
        print("Loss %f" % (total_loss / len(val_loader)))
        # print("Accuracy %f" % acc)
        print("Accuracy %f" % temp_acc)

        print("Sensitivity %f" % sens)
        print("Specificity %f" % spec)
        print("Time (s): " + str(time.time() - time_start))
        print("--------------------------------------")

        # all_test_losses[epoch_id] = total_loss.cpu()
        all_test_losses[epoch_id] = total_loss / len(val_loader)
        all_test_accuracies[epoch_id] = temp_acc

    scheduler.step(val_losses / len(val_loader))

    save_res(epoch_id, total_loss, len(val_loader), temp_acc, sens, spec, time_start, res_name, "val")

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