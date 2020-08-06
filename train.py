
import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from setting import *

# Load the pretrained model from pytorch
vgg16f = models.vgg16(pretrained=True)

# Freeze training for all "features" layers
for param in vgg16f.features.parameters():
    param.requires_grad = False

n_inputs = vgg16f.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

vgg16f.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16f.cuda()

# check to see that your last layer produces the expected number of outputs
print(vgg16f.classifier[6].out_features)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16f.classifier.parameters(), lr=0.001)


# number of epochs to train the model
n_epochs = 2

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16f(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        if batch_i % 20 == 19:  # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0


torch.save(vgg16f.state_dict(), os.path.join(save_dir,'vgg16f.pth'))
torch.cuda.empty_cache()
