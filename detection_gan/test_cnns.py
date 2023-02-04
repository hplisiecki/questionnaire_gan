import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from generator_discriminator import Generator, CNN_Discriminator
import torch
from utils import box_numbers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from data_mixer import data_mixer
from dataloader import real_dataloader
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layer_sizes = [100, 8000, 84000]

generator = Generator(layer_sizes, 40)
discriminator = CNN_Discriminator()
criterion = torch.nn.BCELoss()


generator.to(device)
discriminator.to(device)
batch_size = 100
data = real_dataloader()
real_batches = [(data[0][i:i + batch_size], data[1][i:i + batch_size]) for i in range(0, len(data[0]), batch_size)]

for idx, (inputs, scales) in enumerate(real_batches):
    # zero the parameter gradients

    inputs = torch.tensor(inputs).to(device)
    real_scales = torch.tensor(scales).to(device).to(torch.float32)
    break

z = torch.randn(batch_size, 100, device=device)

# round
fake_data = generator(z, real_scales)
fake_data = fake_data.view(batch_size, 100, 40, -1)
# softmax
fake_data = torch.nn.functional.softmax(fake_data, dim=3)
# get max

mixed_data, labels = data_mixer(inputs, fake_data, device)
mixed_data = mixed_data.to(torch.float32)
labels = labels.to(torch.float32)

discriminator_output = discriminator(mixed_data)
# where_not_padding = torch.where(mixed_data.sum(dim=2).view(-1) > 0)[0]
discriminator_output = discriminator_output.view(-1)
# discriminator_output = discriminator_output[where_not_padding]
labels = labels.view(-1)
# change 2 in labels to 1
labels = torch.where(labels == 2, torch.ones_like(labels), labels)
# labels = labels[where_not_padding]
discriminator_loss = criterion(discriminator_output, labels)




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



import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from dataloader import load_data

epoch = 0
max_scale = 0
for data_file in range(40):
    data_id = data_file % 40
    data_list = load_data(data_id)
    for idx in range(10):
        real_batches = data_list[idx]
        total_loss_disc = 0
        total_loss_gen = 0
        total_disc_acc = 0
        for idx, (inputs, scales) in enumerate(real_batches):
            if scales.max() > max_scale:
                max_scale = scales.max()




from detection_gan.dataloader import real_dataloader
import torch

data = real_dataloader()
batch_size = 100
real_batches = [(data[0][i:i + batch_size], data[1][i:i + batch_size]) for i in range(0, len(data[0]), batch_size)]
for inputs, scales in real_batches:
    inputs = torch.tensor(inputs)

    inputs = torch.argmax(inputs, 3)
    inputs = inputs.view(-1, 40)
    inputs = inputs[inputs.sum(1) != 0]


    cnt = 0
    for ax_ in range(3):
        for ax in range(3):
            data = inputs[i:i+100]
            data = data[:,data.sum(0) != 0]
            length = data.shape[1]
            data = data[:, cnt].cpu().detach().numpy()
            data = data.tolist()
            while data[-1] == 0:
                data = data[:-1]
            cnt += 1
            if cnt == length:
                i+=100
                cnt = 0