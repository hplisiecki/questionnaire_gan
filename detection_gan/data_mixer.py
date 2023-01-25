import torch
import numpy.random as random

def data_mixer(real_data, fake_data, device):
    # mix real and fake data
    ones = torch.ones(real_data.shape[0], real_data.shape[1], 1, device=device)
    zeros = torch.zeros(fake_data.shape[0],  fake_data.shape[1], 1, device=device)
    ones = ones.view(-1, 100, 1)
    zeros = zeros.view(-1, 100, 1)
    real_data = torch.cat((real_data, ones), dim = 2)
    fake_data = torch.cat((fake_data, zeros), dim = 2)
    real_split = torch.split(real_data, 1)
    fake_split = torch.split(fake_data, 1)
    mixed_split = [torch.cat((real_batch, fake_batch), dim = 0) for (real_batch, fake_batch) in zip(real_split, fake_split)]
    mixed_split = [batch.view(-1, 41) for batch in mixed_split]
    mixed_split = [batch[torch.randperm(batch.size()[0])] for batch in mixed_split]
    mixed_split = [batch.view(2, -1, 41) for batch in mixed_split]
    # delete rows that sum to zero
    mixed_split = [torch.cat([torch.cat([half[half[:, :-1].sum(dim = 1) != 0], torch.zeros(len(half[half[:, :-1].sum(dim = 1) == 0]), 41, device = device)])
                              for half in batch]).view(2,-1,41) for batch in mixed_split]
    mixed_data = torch.cat(mixed_split, dim = 0)
    labels = mixed_data[:, :, -1]
    mixed_data = mixed_data[:, :, :-1]
    return mixed_data, labels

'''
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
'''