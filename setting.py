import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# define training and test data directories
save_dir = 'model'
data_dir = 'fruits360/'
train_dir = os.path.join(data_dir, 'Training/')
test_dir = os.path.join(data_dir, 'Test/')
multiple_test_dir = os.path.join(data_dir, 'test_multiple_fruits/')

# define classes
classes = []
path_train = r'fruits360/Training/'
for name in os.listdir(path_train):
    classes.append(name)

# define classes
test_classes = []
path_test = r'fruits360/Test/'
for name in os.listdir(path_test):
    test_classes.append(name)


# load and transform data using ImageFolder

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))

# define dataloader parameters
batch_size = 20
num_workers = 0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True)

