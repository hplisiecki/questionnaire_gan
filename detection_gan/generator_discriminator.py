import torch
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, layer_sizes, scales_size):
        super(Generator, self).__init__()
        layer_sizes[0] = layer_sizes[0] + scales_size
        for idx, layer_dim in enumerate(layer_sizes):
            if idx == len(layer_sizes) - 1:
                continue
            setattr(self, f'linear_{idx}', torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        self.layer_sizes = layer_sizes

    def forward(self, x, scales):
        x = torch.cat((x, scales), dim=1)
        for idx, layer_dim in enumerate(self.layer_sizes):
            if idx == len(self.layer_sizes) - 2:
                break
            x = torch.nn.functional.relu(getattr(self, f'linear_{idx}')(x))
        x = torch.nn.functional.sigmoid(getattr(self, f'linear_{len(self.layer_sizes) - 2}')(x))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, layer_sizes, scales_size):
        layer_sizes[0] = layer_sizes[0] + scales_size
        super(Discriminator, self).__init__()
        for idx, layer_dim in enumerate(layer_sizes):
            if idx == len(layer_sizes) - 1:
                continue
            setattr(self, f'linear_{idx}', torch.nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))

        self.flatten = torch.nn.Flatten()

        self.layer_sizes = layer_sizes

    def forward(self, x, scales):
        x = torch.cat((x, scales), dim=1)
        x = self.flatten(x)
        for idx, layer_dim in enumerate(self.layer_sizes):
            if idx == len(self.layer_sizes) - 2:
                break
            x = torch.nn.functional.relu(getattr(self, f'linear_{idx}')(x))
        x = torch.nn.functional.sigmoid(getattr(self, f'linear_{len(self.layer_sizes) - 2}')(x))
        return x