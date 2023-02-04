import torch
import numpy.random as random

def data_mixer(real_data, fake_data, device):
    # mix real and fake data
    ones = torch.ones(real_data.shape[0], real_data.shape[1], 1, device=device)
    zeros = torch.zeros(fake_data.shape[0],  fake_data.shape[1], 1, device=device, requires_grad= True)
    real_split = torch.split(real_data, 1)
    fake_split = torch.split(fake_data, 1)
    ones_split = torch.split(ones, 1)
    zeros_split = torch.split(zeros, 1)


    mixed_split = [torch.cat((real_batch, fake_batch), dim = 0) for (real_batch, fake_batch) in zip(real_split, fake_split)]
    mixed_split = [batch.view(-1, 40, 21) for batch in mixed_split]

    labels_split = [torch.cat((ones_batch, zeros_batch), dim = 0) for (ones_batch, zeros_batch) in zip(ones_split, zeros_split)]
    labels_split = [batch.view(-1, 1) for batch in labels_split]

    # ones =
    rand_indexes = [torch.randperm(batch.size()[0]).tolist() for batch in mixed_split]

    mixed_split = [batch[batch_indexes] for batch, batch_indexes in zip(mixed_split, rand_indexes)]
    mixed_split = [batch.view(2, -1, 40, 21) for batch in mixed_split]

    labels_split = [batch[batch_indexes] for batch, batch_indexes in zip(labels_split, rand_indexes)]
    labels_split = [batch.view(2, -1, 1) for batch in labels_split]
    # delete rows that sum to zero

    # half = mixed_split[4][0] # shape(100, 40, 20)
    # indexes =  half.sum(dim=1) != 0 # shape (100, 20)
    # half = half[indexes, :, :] # shape (100, 40, 20)
    #
    #
    # [torch.cat([half[half.sum(dim=1) != 0, :, :], torch.zeros(len(half[half[:, :-1].sum(dim=1) == 0]), 40, device=device)])for half in batch]
    mixed_indexes = [torch.cat([torch.argmax(half, 2).sum(dim = 1) != 0 for half in batch]) for batch in mixed_split]
    mixed_indexes = [batch_indexes.view(2, -1) for batch_indexes in mixed_indexes]

    mixed_data = torch.cat([torch.cat([torch.cat([half_data[half_indexes], torch.zeros( (half_data.shape[0] - half_data[half_indexes].shape[0]), 40, 21, device = device )])  for half_data, half_indexes in zip(batch_data, batch_indexes)]).view(2,-1,40,21)
                   for batch_data, batch_indexes in zip(mixed_split, mixed_indexes)], dim = 0)

    labels = torch.cat([torch.cat([torch.cat([half_labels[half_indexes], torch.full( ((half_labels.shape[0] - half_labels[half_indexes].shape[0]), 1), 2, device = device )])  for half_labels, half_indexes in zip(batch_labels, batch_indexes)]).view(2,-1,1)
                     for batch_labels, batch_indexes in zip(labels_split, mixed_indexes)], dim = 0)

    #
    # mixed_data = torch.cat([torch.cat([torch.cat([half[torch.argmax(half, 2).sum(dim = 1) != 0], torch.zeros(len(half[torch.argmax(half, 2).sum(dim = 1) == 0]), 40, 21, device = device)])
    #                 for half in batch]).view(2,-1,40,21) for batch in mixed_split], dim = 0)


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