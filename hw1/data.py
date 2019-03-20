import torch
from torch.utils import data

class DatasetWithoutLabel(data.Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('embed_data/' + ID + '.pt')

        return X


class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('embed_data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y