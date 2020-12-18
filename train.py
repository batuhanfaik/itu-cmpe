import shutil

import numpy as np
import torch
import time
import re
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loader import DataReader, GenericDataReader
import torch.nn as nn
import os
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, models, transforms
from efficientnet import EfficientNet
import resnet
from collections import OrderedDict


def save_res(epoch_id, total_loss, loader_len, acc, time_start, res_name, mode):
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
DATASET_PATH = "/home/ufuk/cassava-leaf-disease-classification"
BATCH_SIZE = 128
num_workers = 1

train_loader = torch.utils.data.DataLoader(
    GenericDataReader(mode='train', fold_name="folds/fold_5_train.txt", path=DATASET_PATH),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    GenericDataReader(mode='val', fold_name="folds/fold_5_val.txt", path=DATASET_PATH),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)

experiment_name = prepare_experiment(experiment_name="baseline_resnet_fold5")
res_name = experiment_name + "/" + experiment_name + "_res.txt"

all_python_files = os.listdir('.')

for i in range(len(all_python_files)):
    if '.py' in all_python_files[i]:
        os.system('cp ' + all_python_files[i] + ' ' + experiment_name + '/code/')

num_classes = 5
num_epochs = 100

model = EfficientNet.from_name("efficientnet-b0", num_classes=num_classes)
model = model.to(device)

lr = 1e-1
base_optimizer = torch.optim.SGD
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-10)
loss = torch.nn.CrossEntropyLoss().to(device)

all_tr_losses = torch.zeros(num_epochs, 1)
all_tr_accuracies = np.zeros((num_epochs, 1))
all_test_losses = torch.zeros(num_epochs, 1)
all_test_accuracies = np.zeros((num_epochs, 1))

for epoch_id in range(1, num_epochs + 1):
    model.train()

    total_loss = 0
    total_true = 0
    total_false = 0
    time_start = time.time()

    for i, data in enumerate(train_loader):
        img = data['image']
        img_class = data['label']
        img = img.to(device)
        img.requires_grad = True
        img_class = img_class.to(device)
        img_class.requires_grad = False
        optimizer.zero_grad()

        output = model(img)
        _, prediction = torch.max(output.data, 1)

        loss_value = loss(output, img_class)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.data
        total_true += torch.sum(prediction == img_class.data)
        total_false += torch.sum(prediction != img_class.data)

        if (i + 1) % 20 == 0:
            print("Pre-report Epoch:", epoch_id)
            print("Loss: %f" % loss_value.data)
            print("Status -> %d / %d" % (i + 1, len(train_loader)))
            print("************************************")

    acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())

    print("Train:", datetime.datetime.now())
    print("Epoch %d scores:" % epoch_id)
    print("Loss: %f" % (total_loss / len(train_loader)))
    print("Accuracy: %f" % acc)
    print("LR: %f" % get_lr(optimizer))
    print("Time (s): " + str(time.time() - time_start))
    print("--------------------------------------")

    # all_tr_losses[epoch_id] = total_loss.cpu()
    all_tr_losses[epoch_id] = total_loss / len(train_loader)
    all_tr_accuracies[epoch_id] = acc

    save_res(epoch_id, total_loss, len(train_loader), acc, time_start, res_name, "train")

    with torch.no_grad():

        model.eval()

        val_losses = 0
        total_loss = 0
        total_true = 0
        total_false = 0
        time_start = time.time()

        for i, batch in enumerate(val_loader):
            img = batch['image']
            img_class = batch['label']
            img = img.to(device)
            img.requires_grad = False

            img_class = img_class.to(device)
            img_class.requires_grad = False
            output = model(img)

            _, prediction = torch.max(output.data, 1)

            val_loss = loss(output, img_class)
            val_losses += val_loss
            total_loss += val_loss.data
            total_true += torch.sum(prediction == img_class.data)
            total_false += torch.sum(prediction != img_class.data)

        acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
        if acc > 0.8:
            model_path = experiment_name + "/models/model_epoch_" + str(epoch_id) + '.pt'
            torch.save(model, model_path)
        print("Test:", datetime.datetime.now())
        print("Val %d scores:" % epoch_id)
        print("Loss %f" % (total_loss / len(val_loader)))
        print("Accuracy %f" % acc)
        print("Time (s): " + str(time.time() - time_start))
        print("--------------------------------------")

        # all_test_losses[epoch_id] = total_loss.cpu()
        all_test_losses[epoch_id] = total_loss / len(val_loader)
        all_test_accuracies[epoch_id] = acc

    scheduler.step(val_losses / len(val_loader))

    save_res(epoch_id, total_loss, len(val_loader), acc, time_start, res_name, "val")

    trainig_loss = all_tr_losses.numpy()
    trainig_loss = np.reshape(trainig_loss, (trainig_loss.shape[1] * trainig_loss.shape[0], -1))

    val_loss = all_test_losses.numpy()
    val_loss = np.reshape(val_loss, (val_loss.shape[1] * val_loss.shape[0], -1))

    trainig_loss[trainig_loss == 0] = np.nan
    val_loss[val_loss == 0] = np.nan

    plt.plot(trainig_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.legend()
    fig_path = experiment_name + "/graphs/train_val_loss_epoch_" + str(epoch_id) + '.png'
    plt.savefig(fig_path)
    plt.clf()
