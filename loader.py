import torch
import glob, os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import pandas as pd


class DataReader(torch.utils.data.Dataset):
	def __init__(self, mode, fold_name):
		super(DataReader, self).__init__()

		self.input_img_paths = []
		self.img_name = []
		self.mode = mode

		df = pd.read_csv("train.csv")
		df_to_dict = df.set_index('image_id').to_dict()
		df_to_dict = df_to_dict['label']

		self.img_label = df_to_dict


		#TODO		
		if mode == 'train':
			self.input_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.5, 0.5, 0.5),
									 std=(0.5, 0.5, 0.5))
			])
		#TODO
		elif mode == 'val':
			self.input_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.5, 0.5, 0.5),
									 std=(0.5, 0.5, 0.5))
			])
		#TODO
		elif mode == 'test':
			self.input_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.5, 0.5, 0.5),
									 std=(0.5, 0.5, 0.5))
			])


		if mode == 'train':
			
			img_name = []
			with open(fold_name) as f:
				img_name = f.readlines()

			img_name = [x.strip() for x in img_name]

			for name in img_name:
				img_path = "train_images/" + name
				self.img_name.append(name)
				self.input_img_paths.append(img_path)


		elif mode == 'val':

			img_name = []
			with open(fold_name) as f:
				img_name = f.readlines()

			img_name = [x.strip() for x in img_name]

			for name in img_name:
				img_path = "train_images/" + name
				self.img_name.append(name)
				self.input_img_paths.append(img_path)

		elif mode == 'test':

			img_name = []
			with open(fold_name) as f:
				img_name = f.readlines()

			img_name = [x.strip() for x in img_name]

			for name in img_name:
				img_path = "train_images/" + name
				self.img_name.append(name)
				self.input_img_paths.append(img_path)

	def load_input_img(self, filepath):
		img = Image.open(filepath).convert('RGB')
		return img

	def __getitem__(self, index):
		if self.mode == 'train':
			img_name = self.img_name[index]
			img = self.load_input_img(self.input_img_paths[index])
			
			label = self.img_label[img_name]
			img = self.input_transform(img)
		
			data = {}
			data['image'] = img
			data['label'] = label

			return data

		elif self.mode == 'val':

			img_name = self.img_name[index]
			img = self.load_input_img(self.input_img_paths[index])
			
			label = self.img_label[img_name]
			img = self.input_transform(img)
		
			data = {}
			data['image'] = img
			data['label'] = label

			return data

		elif self.mode == 'test':
			img_name = self.img_name[index]
			img = self.load_input_img(self.input_img_paths[index])
			
			label = self.img_label[img_name]
			img = self.input_transform(img)
		
			data = {}
			data['image'] = img
			data['label'] = label

			return data

	def __len__(self):
		return len(self.input_img_paths)

"""
BATCH_SIZE = 4
numworkers = 1

train_loader = torch.utils.data.DataLoader(DataReader(mode='train', fold_name="fold_1_train.txt"), batch_size=BATCH_SIZE, shuffle=True,
										   num_workers=numworkers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")										
for i, data in enumerate(train_loader):

	img = data['image']
	img_class = data['label']
	#img = img.to(device)
	
	#print("i: ", i)
	print("img_class = ", img_class)
	print("img = ", img.shape)
	
	print(img_class)
	#if i == 3:
		#break
	
"""
