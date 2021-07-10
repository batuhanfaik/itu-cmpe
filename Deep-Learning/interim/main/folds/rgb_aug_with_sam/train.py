import numpy as np
import torch
import time
import re
import torch.nn.functional as F
#from loader import DataReader
from loader import GenericDataset, DataLoader#new
import torch.nn as nn
import os
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, models, transforms
from efficient import EfficientNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sam import SAM


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


def prepare_experiment(project_path="./rgb_aug/", experiment_name=None):
    next_experiment_number = 0
    for directory in os.listdir(project_path):
        search_result = re.search("experiment_(.*)", directory)
        if search_result and next_experiment_number < int(search_result[1])+1:
            next_experiment_number = int(search_result[1]) + 1

    if not experiment_name:
        experiment_name = "experiment_{}".format(next_experiment_number)
    else:
        experiment_name = experiment_name

    os.mkdir(experiment_name)
    os.mkdir(experiment_name + '/graphs/')
    os.mkdir(experiment_name + '/models/')
    os.mkdir(experiment_name + '/code/')

    return experiment_name



class EfficientSpinalNet(nn.Module):
    def __init__(self, half_in_size, layer_width):
        super(EfficientSpinalNet, self).__init__()

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(layer_width * 4, num_classes), )

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, half_in_size:2 * half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, half_in_size:2 * half_in_size], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/home/ufuk/Desktop/2020_2021_fall/BLG561E/interim_project/Cassava Leaf Disease Classification"
BATCH_SIZE = 12
num_workers = 1

dataset_train = GenericDataset(split="train", fold_name="rgb_aug/folds/fold_2_train.txt", path=DATASET_PATH)
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, unsupervised=True)

dataset_val = GenericDataset(split="val", fold_name="rgb_aug/folds/fold_2_val.txt", path=DATASET_PATH)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, unsupervised=True)


experiment_name = prepare_experiment(experiment_name="rgb_aug/rgb_aug_w_sam_fold2")
#res_name = "rgb_aug/" + experiment_name + "/" + experiment_name + "_res.txt"
res_name = experiment_name + "/" + "rgb_aug_w_sam_fold2" + "_res.txt"

all_python_files = os.listdir('.')

for i in range(len(all_python_files)):
    if '.py' in all_python_files[i]:
        os.system('cp ' + all_python_files[i] + ' ' + experiment_name + '/code/')

num_classes = 30
num_epochs = 30

model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)

spinal_flag = False

if spinal_flag:
    num_features = model.in_channels
    half_in_size = round(num_features / 2)
    layer_width = 20  # Small for Resnet, large for VGG
    model._fc = EfficientSpinalNet(half_in_size=half_in_size, layer_width=layer_width)




model = model.to(device)

lr = 0.00256
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=lr, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-10)
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

    for i, data in enumerate(train_loader()):
        img, img_class = data#new
        #img = data['image']
        #img_class = data['label']
        img = torch.Tensor(img.float())#new
        img = img.to(device)
        img.requires_grad = True
        img_class = img_class.to(device)
        img_class.requires_grad = False
        optimizer.zero_grad()

        output = model(img)
        
        
        _, prediction = torch.max(output.data, 1)

        # First pass
        loss_value = loss(output, img_class)
        loss_value.backward()
        optimizer.first_step(zero_grad=True)

            # Second pass
        loss(model(img), img_class).backward()
        optimizer.second_step(zero_grad=True)  

        #loss_value = loss(output, img_class)
        #loss_value.backward()

        #optimizer.step()

        total_loss += loss_value.data
        total_true += torch.sum(prediction == img_class.data)
        total_false += torch.sum(prediction != img_class.data)
        if (i + 1) % 200 == 0:
            print("Pre-report Epoch:", epoch_id)
            print("Loss: %f" % loss_value.data)
            print("Status -> %d / %d" % (i + 1, len(train_loader())))
            print("************************************")

    acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())

    print("Train:", datetime.datetime.now())
    print("Epoch %d scores:" % epoch_id)
    print("Loss: %f" % (total_loss / len(train_loader())))
    print("Accuracy: %f" % acc)
    print("Time (s): " + str(time.time() - time_start))
    print("--------------------------------------")

    # all_tr_losses[epoch_id] = total_loss.cpu()
    all_tr_losses[epoch_id] = total_loss / len(train_loader())
    all_tr_accuracies[epoch_id] = acc

    save_res(epoch_id, total_loss, len(train_loader()), acc, time_start, res_name, "train")
    
    with torch.no_grad():

        model.eval()

        total_loss = 0
        total_true = 0
        total_false = 0
        time_start = time.time()

        for i, batch in enumerate(val_loader()):
            img, img_class = batch#new
            img = torch.Tensor(img.float())#new
            img = img.to(device)
            img.requires_grad = False

            img_class = img_class.to(device)
            img_class.requires_grad = False
            output = model(img)
            
            
            temp_list = []

            temp_list = torch.cat([output[:, 0:6].sum(dim=1, keepdim=True), output[:, 6:12].sum(dim=1, keepdim=True), output[:, 12:18].sum(dim=1, keepdim=True), output[:, 18:24].sum(dim=1, keepdim=True), 
                        output[:, 24:30].sum(dim=1, keepdim=True)],dim=1) 
            new_output = temp_list
            temp_list_class = []

            
            temp_list_class = img_class//6
            new_img_class = temp_list_class
            _, prediction = torch.max(new_output.data, 1)

            total_loss += loss(new_output, new_img_class).data
            total_true += torch.sum(prediction == new_img_class.data)
            total_false += torch.sum(prediction != new_img_class.data)

        acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
        if acc > 0.6:
            model_path = experiment_name + "/models/model_epoch_" + str(epoch_id) + '.pt'
            torch.save(model, model_path)
        print("Test:", datetime.datetime.now())
        print("Val %d scores:" % epoch_id)
        print("Loss %f" % (total_loss / len(val_loader())))
        print("Accuracy %f" % acc)
        print("Time (s): " + str(time.time() - time_start))
        print("--------------------------------------")

        # all_test_losses[epoch_id] = total_loss.cpu()
        all_test_losses[epoch_id] = total_loss / len(val_loader())
        all_test_accuracies[epoch_id] = acc
    scheduler.step(total_loss / len(val_loader()))
    save_res(epoch_id, total_loss, len(val_loader()), acc, time_start, res_name, "val")

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
    