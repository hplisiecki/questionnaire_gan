import numpy as np
from torch import nn
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.training = False
        if 'label' in df.columns:
            self.labels = df['label'].values.astype(float)
            self.training = True

        self.vectors = [torch.tensor(text) for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.vectors)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_vectors(self, idx):
        # Fetch a batch of inputs
        return self.vectors[idx]

    def __getitem__(self, idx):

        batch_vectors = self.get_batch_vectors(idx)
        if self.training:
            batch_labels = self.get_batch_labels(idx)
            return batch_vectors, batch_labels

        return batch_vectors

class Simple_Net(nn.Module):

    def __init__(self, dropout=0.1, input_size = 300, layer_sizes = []):

        super(Simple_Net, self).__init__()

        self.layer_sizes = layer_sizes

        self.start_layer = nn.Linear(input_size, layer_sizes[0])

        for idx, layer_dim in enumerate(layer_sizes):
            if idx == 0:
                continue
            setattr(self, f'linear_{idx}', nn.Linear(layer_sizes[idx-1], layer_sizes[idx]))

        self.final = nn.Linear(layer_sizes[-1], 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # sigmoid


    def forward(self, x):

        x = self.start_layer(x)

        for idx, layer_dim in enumerate(layer_sizes):
            if idx == 0:
                continue
            x = self.dropout(x)
            x = getattr(self, f'linear_{idx}')(x)
            x = self.relu(x)

        x = self.final(x)
        x = self.relu(x)

        return x