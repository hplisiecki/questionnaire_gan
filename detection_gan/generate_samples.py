import sys
sys.path.insert(0, r'D:\GitHub\questionnaire_gan\detection_gan')
from generator_discriminator import Generator, Discriminator
import torch
from utils import box_numbers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from data_mixer import data_mixer
from dataloader import real_dataloader
import numpy as np


# layer_sizes = [100, 500, 1000, 2000, 4000] 1

# layer_sizes = [100, 500, 1000, 2000, 3000, 4000] 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layer_sizes = [100, 4000]

generator = Generator(layer_sizes, 40)
layer_sizes = [4000, 3000, 2000, 1000, 500, 100]
discriminator = Discriminator(layer_sizes, 40)



# load
generator.load_state_dict(torch.load(r'D:\GitHub\questionnaire_gan\detection_gan\models\test_3generator'))
discriminator.load_state_dict(torch.load(r'D:\GitHub\questionnaire_gan\detection_gan\models\test_3discriminator'))
discriminator.to(device)
generator.to(device)
batch_size = 100
real_batches = real_dataloader(batch_size)

batch = real_batches[0]
inputs = torch.tensor(batch[0]).to(device)
real_scales = torch.tensor(batch[1]).to(device).to(torch.float32)
z = torch.randn(batch_size, 100).to(device)
scales = [3] * 40
scales = torch.Tensor(scales).to(device).view(1,-1)
scales = torch.cat([scales for _ in range(batch_size)], dim = 0)
fake_data = generator(z, scales)

new_scale = torch.cat([torch.flatten(i.repeat(100, 1)) for i in scales]).to(device)

fake_data = box_numbers(torch.flatten(fake_data), new_scale)
fake_data = fake_data.view(batch_size, -1, 40)
real_data = inputs
mixed_data, labels = data_mixer(inputs, fake_data, device)
mixed_data = mixed_data.to(torch.float32)
double_scales = torch.cat((real_scales, real_scales), dim=0).view(-1, 1, 40)

discriminator_output = discriminator(mixed_data, double_scales)

flat_labels = labels.view(-1)
flat_preds = discriminator_output.view(-1)
flat_preds = torch.round(flat_preds)

print(len([1 for (i, j)  in zip(flat_preds, flat_labels) if i == j]) / len(flat_labels))