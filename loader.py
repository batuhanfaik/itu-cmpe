import torch
import glob, os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import pandas as pd


def load_input_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DataReader(torch.utils.data.Dataset):
    def __init__(self, mode, fold_name, path=""):
        super(DataReader, self).__init__()

        self.input_img_paths = []
        self.img_name = []
        self.mode = mode
        self.path = path

        df = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_to_dict = df.set_index('image_id').to_dict()
        df_to_dict = df_to_dict['label']

        self.img_label = df_to_dict

        # TODO
        if mode == 'train':
            self.input_transform = transforms.Compose([
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])
        # TODO
        elif mode == 'val':
            self.input_transform = transforms.Compose([
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])
        # TODO
        elif mode == 'test':
            self.input_transform = transforms.Compose([
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])

        if mode == 'train':
            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)
        elif mode == 'val':
            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)

        elif mode == 'test':
            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_name = self.img_name[index]
            img = load_input_img(self.input_img_paths[index])

            label = self.img_label[img_name]
            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

        elif self.mode == 'val':
            img_name = self.img_name[index]
            img = load_input_img(self.input_img_paths[index])

            label = self.img_label[img_name]
            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

        elif self.mode == 'test':
            img_name = self.img_name[index]
            img = load_input_img(self.input_img_paths[index])

            label = self.img_label[img_name]
            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

    def __len__(self):
        return len(self.input_img_paths)


class GenericDataReader(data.Dataset):
    def __init__(self, mode="train", random_sized_crop=False,
                 fold_name="", path=""):

        self.random_sized_crop = random_sized_crop
        self.mean_pix = [0.5, 0.5, 0.5]
        self.std_pix = [0.5, 0.5, 0.5]
        self.input_img_paths = []
        self.img_name = []
        self.split = mode
        self.path = path

        df = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_to_dict = df.set_index('image_id').to_dict()
        df_to_dict = df_to_dict['label']

        self.img_label = df_to_dict
        transforms_list = []
        if self.split == 'train':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]

            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)

        elif self.split == 'val':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]

            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)

        elif self.split == 'test':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]

            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)

        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        img = load_input_img(self.input_img_paths[index])
        label = self.img_label[img_name]
        img = self.transform(img)
        return {'image': img, 'label': label}

    def __len__(self):
        return len(self.input_img_paths)
