import torch
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, layer_sizes, scales_size):
        super(Generator, self).__init__()
        layer_sizes[0] = layer_sizes[0] + scales_size
        for idx, layer_dim in enumerate(layer_sizes):
            if idx == len(layer_sizes) - 1:
                continue
            setattr(self, f'linear_gen_{idx}', torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        self.layer_sizes = layer_sizes

    def forward(self, x, scales):
        x = torch.cat((x, scales), dim=1)
        for idx, layer_dim in enumerate(self.layer_sizes):
            if idx == len(self.layer_sizes) - 2:
                break
            x = torch.nn.functional.relu(getattr(self, f'linear_gen_{idx}')(x))
        x = torch.relu(getattr(self, f'linear_gen_{len(self.layer_sizes) - 2}')(x))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, layer_sizes, scales_size):
        layer_sizes[0] = layer_sizes[0] + scales_size
        super(Discriminator, self).__init__()
        for idx, layer_dim in enumerate(layer_sizes):
            if idx == len(layer_sizes) - 1:
                continue
            setattr(self, f'linear_disc_{idx}', torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))


        self.layer_sizes = layer_sizes

    def forward(self, x, scales):
        x = torch.cat((x, scales), dim=1)
        x = x.view(-1)
        for idx, layer_dim in enumerate(self.layer_sizes):
            if idx == len(self.layer_sizes) - 2:
                break
            x = torch.nn.functional.relu(getattr(self, f'linear_disc_{idx}')(x))
        x = torch.sigmoid(getattr(self, f'linear_{len(self.layer_sizes) - 2}')(x))
        return x
class CNN_Generator(torch.nn.Module):
    def __init__(self):
        super(CNN_Generator, self).__init__()

        self.model = torch.nn.Sequential()
        for i in range(13):
            self.model.add_module(f'conv_gen_{i}', torch.nn.ConvTranspose1d(in_channels=100, out_channels=100, kernel_size=4, stride=1))
            self.model.add_module(f'norm_gen_{i}', torch.nn.BatchNorm1d(100))
            self.model.add_module(f'relu_gen_{i}', torch.nn.ReLU())


    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class CNN_Discriminator(torch.nn.Module):
    def __init__(self):
        super(CNN_Discriminator, self).__init__()
        self.conv_disc_simple = torch.nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(1,21), stride=1) # 40
        self.conv_disc_short = torch.nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(3,21), stride=1) # 38
        self.conv_disc_long = torch.nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(6,21), stride=1) # 35
        self.linear_disc_1 = torch.nn.Linear(((35 * 100) + (38 * 100) + (40 * 100)), 300)
        self.linear_disc_2 = torch.nn.Linear(300, 100)
        self.norm = torch.nn.BatchNorm2d(100)


    def forward(self, x):
        x_simple = torch.nn.functional.relu(self.conv_disc_simple(x))
        x_short = torch.nn.functional.relu(self.conv_disc_short(x))
        x_long = torch.nn.functional.relu(self.conv_disc_long(x))
        x_simple = self.norm(x_simple)
        x_short = self.norm(x_short)
        x_long = self.norm(x_long)
        x = torch.cat((x_simple, x_short, x_long), dim=2)
        x = x.squeeze()
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.linear_disc_1(x))
        x = self.linear_disc_2(x)
        x = torch.sigmoid(x)

        return x
