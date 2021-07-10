import torch
import glob, os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from torch.utils.data.dataloader import default_collate
import torchnet as tnt
import torch.utils.data as data

def load_input_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
class GenericDataset(data.Dataset):
    def __init__(self, split="train", random_sized_crop=False,
                 fold_name="", path=""):


        self.random_sized_crop = random_sized_crop
        self.mean_pix = [0.5, 0.5, 0.5]
        self.std_pix = [0.5, 0.5, 0.5]
        self.input_img_paths = []
        self.img_name = []
        self.split = split
        self.path = path

        df = pd.read_csv(os.path.join(self.path, "train.csv"))
        df_to_dict = df.set_index('image_id').to_dict()
        df_to_dict = df_to_dict['label']

        self.img_label = df_to_dict
        transforms_list =[]
        if self.split=='train':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                lambda x: np.asarray(x),
            ]

            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)
        
        elif self.split=='val':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                lambda x: np.asarray(x),
            ]

            with open(fold_name) as f:
                img_name = f.readlines()
            img_name = [x.strip() for x in img_name]

            for name in img_name:
                img_path = os.path.join(self.path, "train_images", name)
                self.img_name.append(name)
                self.input_img_paths.append(img_path)
        
        elif self.split=='test':
            transforms_list = [
                transforms.Resize(size=(300, 225), interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                lambda x: np.asarray(x),
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
        label = self.img_label[img_name] * 6
        img = self.transform(img)
        return img, label , img_name       

    def __len__(self):
        return len(self.input_img_paths)


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=False,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = [0.5, 0.5, 0.5]
        std_pix   = [0.5, 0.5, 0.5]
        self.transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:

            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, img0_label, img0_name = self.dataset[idx]
                r = img0[:,:,0]
                g = img0[:,:,1]
                b = img0[:,:,2]
                rotated_imgs = [
                    torch.from_numpy(np.stack((r,g,b),2)),
                    torch.from_numpy(np.stack((r,b,g),2)),
                    torch.from_numpy(np.stack((g,r,b),2)),
                    torch.from_numpy(np.stack((g,b,r),2)),
                    torch.from_numpy(np.stack((b,r,g),2)),
                    torch.from_numpy(np.stack((b,g,r),2))
                ]
                rotation_labels = torch.LongTensor([img0_label, img0_label + 1, img0_label + 2, img0_label + 3, img0_label + 4, img0_label + 5])
                #print("name: ", img0_name, " label: ", img0_label)
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, width, height, channels = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle, drop_last=True)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
