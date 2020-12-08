import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from setting import *

vgg19f = models.vgg19(pretrained=True)

# Freeze training for all "features" layers
for param in vgg19f.features.parameters():
    param.requires_grad = False

n_inputs = vgg19f.classifier[6].in_features

# add last linear layer
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

vgg19f.classifier[6] = last_layer

if train_on_gpu:
    vgg19f.cuda()

vgg19f.load_state_dict(torch.load(os.path.join(save_dir, 'vgg19f.pth')))

vgg19f.eval() # eval mode

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
# optimizer = optim.SGD(vgg19f.classifier.parameters(), lr=0.001)
optimizer = optim.SGD(vgg19f.classifier.parameters(), lr=0.001, momentum=0.9)

# track test loss
# over n fruits
Num_fruits = len(test_classes)
test_loss = 0.0
class_correct = list(0. for i in range(Num_fruits))
class_total = list(0. for i in range(Num_fruits))


# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg19f(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(Num_fruits):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
