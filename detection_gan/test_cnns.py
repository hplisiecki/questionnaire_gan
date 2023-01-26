import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from generator_discriminator import CNN_Generator
import torch
from utils import box_numbers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from data_mixer import data_mixer
from dataloader import real_dataloader
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layer_sizes = [100, 500, 1000, 2000, 3000, 4000]

generator = CNN_Generator(layer_sizes, 40)

generator.to(device)
batch_size = 100
real_batches = real_dataloader(batch_size)

batch = real_batches[0]
inputs = torch.tensor(batch[0]).to(device)
real_scales = torch.tensor(batch[1]).to(device).to(torch.float32)


mixed_data, labels = data_mixer(inputs, inputs, device)
mixed_data = mixed_data.to(torch.float32)
double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)

x = torch.cat((mixed_data, double_scales), dim=1)
print(x.shape)

layer = torch.nn.Conv1d(in_channels=100, out_channels=100, kernel_size=1, stride=1)
layer.to(device)

# batch_norm
norm = torch.nn.BatchNorm1d(200)
norm.to(device)
y = layer(mixed_data)
print(y.shape)

flatten = torch.nn.Flatten()
linear_1 = torch.nn.Linear(4000, 200)
flatten.to(device)
linear_1.to(device)

z = flatten(mixed_data)
z = linear_1(z)
print(z.shape)

# concat z and y
y = y.view(200, )





z = torch.randn(batch_size, 100).to(device)
scales = [3] * 20
scales.extend([0] * 20)
scales = torch.Tensor(scales).to(device).view(1,-1)
scales = torch.cat([scales for _ in range(batch_size)], dim = 0)

# x = torch.cat((z, scales), dim=1)
x = z
x = x.view(batch_size, -1, 1)
print(x.shape)

layer = torch.nn.ConvTranspose1d(100, 100, 4, stride=1)
layer.to(device)

a = layer(x)
print(a.shape)

layer = torch.nn.ConvTranspose1d(100, 100, 4, stride=1)
layer.to(device)
a = layer(a)
print(a.shape)

fake_data = generator(z, scales)


model = torch.nn.Sequential()
for i in range(13):
    model.append(torch.nn.ConvTranspose1d(in_channels=100, out_channels=100, kernel_size=4, stride=1))

model.to(device)

a = model(x)
print(a.shape)