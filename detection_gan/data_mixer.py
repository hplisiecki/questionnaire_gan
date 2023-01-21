import torch
import numpy.random as random

def data_mixer(real_data, fake_data):
    # mix real and fake data
    ones = torch.ones(real_data.shape[0])
    zeros = torch.zeros(fake_data.shape[0])
    labels = torch.cat((ones, zeros), 0)
    mixed_data = torch.cat((real_data, fake_data), dim=0)
    labels = labels.view(-1, 1)
    mixed_data = torch.cat((mixed_data, labels), dim=1)
    # shuffle
    mixed_data = mixed_data[torch.randperm(mixed_data.size()[0])]
    labels = mixed_data[:, -1]
    mixed_data = mixed_data[:, :-1]
    return mixed_data, labels

real_data = torch.randn((5, 5))
fake_data = torch.randn((5, 5))
labels = torch.tensor([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
mixed_data, labelsNew = data_mixer(real_data, fake_data)
#
print(real_data)
print(fake_data)
print(labels)
print(mixed_data)
print(labelsNew)