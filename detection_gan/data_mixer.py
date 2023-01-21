import torch
import numpy.random as random

def data_mixer(real_data, fake_data, labels):
    # mix real and fake data
    mixed_data = torch.cat((real_data, fake_data), dim=0)
    mixed_data = torch.cat((mixed_data, labels), dim1=1)
    mixed_data = random.shuffle(mixed_data, lambda: random.poisson(2, mixed_data.shape[0]))
    labels = torch.split(mixed_data, )
    
    return mixed_data, labels

real_data = torch.randn((10, 10))
fake_data = torch.randn((10, 10))
labels = torch.tensor([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

mixed_data, labelsNew = data_mixer(real_data, fake_data, labels)

print(real_data)
print(fake_data)
print(labels)
print(mixed_data)
print(labelsNew)