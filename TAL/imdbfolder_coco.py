# imdbfolder_coco.py
# Created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details
#
# This file is used for loading and preprocessing COCO-format datasets, suitable for multi-task learning scenarios such as Decathlon.

import os
import os.path
import pickle
import numpy as np

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

import config_task

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')


class Noise:
    def __init__(self, noise_std=0.08):
        """
        Add Gaussian noise to the image.
        :param noise_std: Standard deviation of the noise, default is 0.08 (recommended for CIFAR-10).
        """
        self.noise_std = noise_std
        print('evaluate on noisy images with std={:.2f}'.format(noise_std))

    def __call__(self, img):
        # Add Gaussian noise and clip pixel values to [0, 1]
        img = torch.clip(img + torch.randn(size=img.shape) * self.noise_std, 0, 1)
        return img



class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, index=None,
            labels=None, imgs=None, loader=pil_loader, skip_label_indexing=0):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        if index is not None:
            imgs = [imgs[i] for i in index]
        self.imgs = imgs
        if index is not None:
            if skip_label_indexing == 0:
                labels = [labels[i] for i in index]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index][0]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def prepare_data_loaders(
    dataset_names,
    batch_size=128,
    data_dir='./data/decathlon/',  # Use relative path as default
    imdb_dir='./data/decathlon/annotations',
    shuffle_train=True,
    index=None
):
    train_loaders = []
    val_loaders = []
    num_classes = []
    train = [0]
    val = [1]
    config_task.offset = []

    imdb_names_train = [imdb_dir + '/' + dataset_names[i] + '_train.json' for i in range(len(dataset_names))]
    imdb_names_val   = [imdb_dir + '/' + dataset_names[i] + '_val.json' for i in range(len(dataset_names))]
    imdb_names = [imdb_names_train, imdb_names_val]

    with open(data_dir + 'decathlon_mean_std.pickle', 'rb') as handle:
        # dict_mean_std = pickle.load(handle)
        dict_mean_std = pickle.load(handle, encoding='latin1')
    
    for i in range(len(dataset_names)):
        imgnames_train = []
        imgnames_val = []
        labels_train = []
        labels_val = []
        for itera1 in train+val:
            annFile = imdb_names[itera1][i]
            coco = COCO(annFile)
            imgIds = coco.getImgIds()
            annIds = coco.getAnnIds(imgIds=imgIds)
            anno = coco.loadAnns(annIds)
            images = coco.loadImgs(imgIds) 
            timgnames = [img['file_name'] for img in images]
            timgnames_id = [img['id'] for img in images]
            labels = [int(ann['category_id'])-1 for ann in anno]
            min_lab = min(labels)
            labels = [lab - min_lab for lab in labels]
            max_lab = max(labels)

            imgnames = []
            for j in range(len(timgnames)):
                
                
                # Remove unnecessary 'data/' prefix
                corrected_path = timgnames[j].replace('data/', '')
                if dataset_names[i] in ['imagenet']:
                    corrected_path = corrected_path.replace('imagenet12', 'imagenet')
                imgnames.append((os.path.join(data_dir, corrected_path), timgnames_id[j]))
                
                            
                # imgnames.append((data_dir + '/' + timgnames[j],timgnames_id[j]))

            if itera1 in train:
                imgnames_train += imgnames
                labels_train += labels
            if itera1 in val:
                imgnames_val += imgnames
                labels_val += labels

        num_classes.append(int(max_lab+1))
        config_task.offset.append(min_lab)
        if dataset_names[i] in ['imagenet']:
            
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        else:
            means = dict_mean_std[dataset_names[i] + 'mean']
            stds = dict_mean_std[dataset_names[i] + 'std']

        if dataset_names[i] in ['gtsrb', 'omniglot', 'svhn']:  # no horizontal flip
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                Noise(noise_std=0.05),
                transforms.Normalize(means, stds),
            ])
        else:  # Other datasets, including horizontal flip and color augmentation
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                transforms.ToTensor(),
                Noise(noise_std=0.08),  # Add noise augmentation for other datasets
                transforms.Normalize(means, stds),
            ])
            
        if dataset_names[i] in ['gtsrb', 'omniglot', 'svhn']:  # no horizontal flip needed
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
            
            
        img_path = data_dir
        trainloader = torch.utils.data.DataLoader(ImageFolder(data_dir, transform_train, None, index, labels_train, imgnames_train), batch_size=batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True)
        valloader = torch.utils.data.DataLoader(ImageFolder(data_dir, transform_test, None, None, labels_val, imgnames_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        train_loaders.append(trainloader)
        val_loaders.append(valloader) 
    
    return train_loaders, val_loaders, num_classes

