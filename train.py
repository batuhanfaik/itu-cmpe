import numpy as np
import torch
import time
import torch.nn.functional as F
from loader import DataReader
import torch.nn as nn
import os
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, models, transforms
from efficient import EfficientNet
from collections import OrderedDict
from model import Net



def save_res(epoch_id,total_loss,loader_len,acc,time_start,res_name,mode):

	with open(res_name, "a") as f:
		f.write(mode)
		f.write(": ")
		f.write(str(datetime.datetime.now()))
		f.write("\n")

		f.write("Epoch ")
		#f.write(str(i))
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





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 8
numworkers = 1

train_loader = torch.utils.data.DataLoader(DataReader(mode='train', fold_name="folds/fold_1_train.txt"), batch_size=BATCH_SIZE, shuffle=True,
										   num_workers=numworkers, drop_last=True)
val_loader = torch.utils.data.DataLoader(DataReader(mode='val', fold_name="folds/fold_1_test.txt"), batch_size=BATCH_SIZE, shuffle=True,
										  num_workers=numworkers, drop_last=True)


experiment_name = "experiment_1"
res_name = experiment_name + "/" + experiment_name + "_res.txt"

os.mkdir(experiment_name)
os.mkdir(experiment_name + '/graphs/')
os.mkdir(experiment_name + '/models/')
os.mkdir(experiment_name + '/code/')


all_python_files = os.listdir('.')


for i in range(len(all_python_files)):
	if '.py' in all_python_files[i]:
		os.system('cp '+ all_python_files[i]+ ' ' + experiment_name + '/code/')

num_classes = 5
num_epochs = 100

model = EfficientNet.from_name('efficientnet-b0')


# Set whether to freeze model parameters (=False : Freeze)
"""
for param in model.parameters():
	param.requires_grad = True
"""

"""
# Add fully connected classifier
classifier = nn.Sequential(OrderedDict([
	#("fc1", nn.Linear(2560, 1024)),
	#("fc1", nn.Linear(1408, 1024)),
	("fc1", nn.Linear(4, 1280)),
	("relu1", nn.ReLU()),
	("dropout1", nn.Dropout(0.5)),
	("fc2", nn.Linear(1280, 512)),
	("relu2", nn.ReLU()),
	("dropout2", nn.Dropout(0.5)),
	("fc3", nn.Linear(512, 256)),
	("relu3", nn.ReLU()),
	("dropout3", nn.Dropout(0.5)),
	("fc4", nn.Linear(256, num_classes))
	# ("out", nn.LogSoftmax(dim=1))
]))

model._fc = classifier
"""

model = model.to(device)
lr = 0.00256
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)


loss = torch.nn.CrossEntropyLoss().to(device)


all_tr_losses = np.zeros((num_epochs, 1))
all_tr_accuracies = np.zeros((num_epochs, 1))
all_test_losses = np.zeros((num_epochs, 1))
all_test_accuracies = np.zeros((num_epochs, 1))

for epoch_id in range(1, num_epochs + 1):

	model.train()

	if epoch_id % 20 == 0:
		for param_group in optimizer.param_groups:
			param_group["lr"] = param_group["lr"] / 1.5

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

		if (i + 1) % 200 == 0:
			print("Pre-report Epoch:", epoch_id)
			print("Loss: %f" % loss_value.data)
			print("Status -> %d / %d" % (i + 1, len(train_loader)))
			print("************************************")

	acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())


	print("Train:", datetime.datetime.now())
	print("Epoch %d scores:" % epoch_id)
	print("Loss: %f" % (total_loss / len(train_loader)))
	print("Accuracy: %f" % acc)
	print("Time (s): " + str(time.time() - time_start))
	print("--------------------------------------")

	all_tr_losses[epoch_id] = total_loss.cpu()
	all_tr_accuracies[epoch_id] = acc

	save_res(epoch_id,total_loss,len(train_loader),acc,time_start,res_name,"train")
	
	with torch.no_grad():

		model.eval()

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

			total_loss += loss(output, img_class).data
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

		all_test_losses[epoch_id] = total_loss.cpu()
		all_test_accuracies[epoch_id] = acc

	save_res(epoch_id,total_loss,len(val_loader),acc,time_start,res_name,"val")

	all_tr_losses[all_tr_losses == 0] = np.nan
	all_test_losses[all_test_losses == 0] = np.nan



	trainig_loss = np.reshape(all_tr_losses, (all_tr_losses.shape[1] * all_tr_losses.shape[0], -1))
	val_loss = np.reshape(all_test_losses, (all_test_losses.shape[1] * all_test_losses.shape[0], -1))

	plt.plot(trainig_loss, label='Train')
	plt.plot(val_loss, label='Validation')
	plt.legend()
	fig_path = experiment_name + "/graphs/train_val_loss_epoch_" + str(epoch_id) + '.png'
	plt.savefig(fig_path)
	plt.clf()
