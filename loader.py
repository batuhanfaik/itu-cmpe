import torch
import glob, os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import cv2
from skimage import exposure
import pandas as pd

np.random.seed(1773)


def load_input_img(filepath, crx_norm):
    # Apply CRX normalization
    if crx_norm:
        # Read grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=crx_norm["clip_limit"], tileGridSize=crx_norm["tile_grid_size"])
        img = clahe.apply(img)
        # Median Filtering
        img = cv2.medianBlur(img, crx_norm["median_filter_size"])
        # Contrast Stretching
        lower_percentile = np.percentile(img, crx_norm["percentiles"][0])
        upper_percentile = np.percentile(img, crx_norm["percentiles"][1])
        img = exposure.rescale_intensity(img, in_range=(lower_percentile, upper_percentile))
        # Back to PIL
        img = Image.fromarray(img)
    else:
        img = Image.open(filepath).convert('L')
    return img


class DataReader(torch.utils.data.Dataset):
    def __init__(self, mode, path, oversample, multi_class, dataset_path=None, crx_norm=None):
        super(DataReader, self).__init__()

        self.input_img_paths = []
        self.input_label = []
        self.mode = mode
        self.multi_class = multi_class
        self.dataset_path = dataset_path
        self.crx_norm = crx_norm

        df = pd.read_csv(path)
        df = df.fillna(0)
        df_to_dict = df.set_index('X_ray_image_name').to_dict()

        self.label = df_to_dict['Label']
        self.dataset_type = df_to_dict['Dataset_type']
        self.label_1 = df_to_dict['Label_1_Virus_category']
        self.label_2 = df_to_dict['Label_2_Virus_category']

        self.healthy = []

        if mode == 'train':
            self.input_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(224, 224), interpolation=Image.BILINEAR),
                transforms.RandomPerspective(distortion_scale=0.1, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     #std=(0.5, 0.5, 0.5))
            ])
        # TODO
        elif mode == 'val':
            self.input_transform = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     #std=(0.5, 0.5, 0.5))
            ])
        # TODO
        elif mode == 'test':
            self.input_transform = transforms.Compose([
                transforms.Resize(size=(224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     #std=(0.5, 0.5, 0.5))
            ])

        if mode == 'train':

            # train_input_folder = os.listdir("train")
            train_input_folder = df['X_ray_image_name']
            train_input_folder = sorted(train_input_folder)

            normal_counter = 0
            corona_counter = 0

            for i in range(len(train_input_folder)):
                if self.dataset_type[train_input_folder[i]] == "TRAIN":
                    temp_path = "train/" + train_input_folder[i]  # input image's path
                    if self.dataset_path:
                        temp_path = os.path.join(self.dataset_path, temp_path)
                    self.input_img_paths.append(temp_path)

                    if multi_class == True:
                        if self.label_1[train_input_folder[i]] == "Virus":
                            self.input_label.append(1)
                            corona_counter += 1
                        elif self.label_1[train_input_folder[i]] == "bacteria":
                            self.input_label.append(2)
                            corona_counter += 1
                        elif self.label_1[train_input_folder[i]] == "Stress-Smoking":
                            self.input_label.append(3)
                            corona_counter += 1
                        else:
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                            normal_counter += 1

                    else:
                        if self.label[train_input_folder[i]] == "Normal":
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                            normal_counter += 1
                        else:
                            self.input_label.append(1)
                            corona_counter += 1

            if oversample == True:
                num_size = corona_counter - normal_counter
                random_numbers = np.random.randint(len(self.healthy), size=num_size)

                for i in random_numbers:
                    self.input_img_paths.append(self.healthy[i])
                    self.input_label.append(0)
                    normal_counter += 1

        elif mode == 'val':

            # val_input_folder = os.listdir("train")
            val_input_folder = df['X_ray_image_name']
            val_input_folder = sorted(val_input_folder)

            normal_counter = 0
            corona_counter = 0

            for i in range(len(val_input_folder)):
                if self.dataset_type[val_input_folder[i]] == "VAL":
                    temp_path = "train/" + val_input_folder[i]  # input image's path
                    if self.dataset_path:
                        temp_path = os.path.join(self.dataset_path, temp_path)
                    self.input_img_paths.append(temp_path)

                    if multi_class == True:
                        if self.label_1[val_input_folder[i]] == "Virus":
                            self.input_label.append(1)
                            corona_counter += 1
                        elif self.label_1[val_input_folder[i]] == "bacteria":
                            self.input_label.append(2)
                            corona_counter += 1
                        elif self.label_1[val_input_folder[i]] == "Stress-Smoking":
                            self.input_label.append(3)
                            corona_counter += 1
                        else:
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                            normal_counter += 1

                    else:
                        if self.label[val_input_folder[i]] == "Normal":
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                        else:
                            self.input_label.append(1)

        elif mode == 'test':
            # test_input_folder = os.listdir("test")
            test_input_folder = df['X_ray_image_name']
            test_input_folder = sorted(test_input_folder)

            normal_counter = 0
            corona_counter = 0

            for i in range(len(test_input_folder)):
                if self.dataset_type[test_input_folder[i]] == "TEST":
                    temp_path = "test/" + test_input_folder[i]  # input image's path
                    if self.dataset_path:
                        temp_path = os.path.join(self.dataset_path, temp_path)
                    self.input_img_paths.append(temp_path)

                    if multi_class == True:
                        if self.label_1[test_input_folder[i]] == "Virus":
                            self.input_label.append(1)
                            corona_counter += 1
                        elif self.label_1[test_input_folder[i]] == "bacteria":
                            self.input_label.append(2)
                            corona_counter += 1
                        elif self.label_1[test_input_folder[i]] == "Stress-Smoking":
                            self.input_label.append(3)
                            corona_counter += 1
                        else:
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                            normal_counter += 1

                    else:
                        if self.label[test_input_folder[i]] == "Normal":
                            self.input_label.append(0)
                            self.healthy.append(temp_path)
                        else:
                            self.input_label.append(1)

    def __getitem__(self, index):
        if self.mode == 'train':
            img = load_input_img(self.input_img_paths[index], self.crx_norm)
            label = self.input_label[index]

            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

        elif self.mode == 'val':
            img = load_input_img(self.input_img_paths[index], self.crx_norm)
            label = self.input_label[index]

            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

        elif self.mode == 'test':
            img = load_input_img(self.input_img_paths[index], self.crx_norm)
            label = self.input_label[index]

            img = self.input_transform(img)

            data = {'image': img, 'label': label}

            return data

    def __len__(self):
        return len(self.input_img_paths)
