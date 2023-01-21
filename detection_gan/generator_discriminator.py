import torch
import numpy as np

# generator and discriminator

class Generator(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(Generator, self).__init__()
        for idx, layer_dim in enumerate(layer_sizes):
            if idx == 0:
                continue
            setattr(self, f'linear_{idx}', torch.nn.Linear(layer_sizes[idx-1], layer_sizes[idx]))
        self.layer_sizes = layer_sizes
        self.input_size = layer_sizes[0]
    
    def forward(self, x, scales):
        scalesM = torch.zeros(self.input_size)
        scalesM[:scales.shape[0], :scales.shape[1]] = scales
        x = torch.cat((x, scalesM), axis=0)
        for idx, layer_dim in enumerate(self.layer_sizes[:-1]):
            if idx == 0:
                continue
            x = torch.nn.functional.relu(getattr(self, f'linear_{idx}')(x))
        x = torch.nn.functional.sigmoid(getattr(self, f'linear_{len(self.layer_sizes)}')(x))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.sigmoid(self.linear3(x))
        return x