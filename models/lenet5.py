from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import torch


class LeNet5(nn.Module):
   def __init__(self, num_classes, channel=1, multi_outputs=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.multi_outputs = multi_outputs

   def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.multi_outputs:
            inter_x = x
            x = self.fc3(x)
            return inter_x, x
        else:
            x = self.fc3(x)
            return x

   def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LayerLeNet5(nn.Module):
   def __init__(self, num_classes):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, num_classes)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       fc1 = x
       x = F.relu(self.fc2(x))
       fc2 = x
       x = self.fc3(x)
       fc3 = x
       return fc1, fc2, fc3

   def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def lenet5(num_classes=10, dataset='mnist', multi_outputs=False):
    if dataset == 'mnist':
        model = LeNet5(num_classes)
    else:
        model = LeNet5(num_classes, 3)
    return model

def layer_lenet5(num_classes=10):
    model = LayerLeNet5(num_classes)
    return model
