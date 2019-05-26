import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader

# from IPython import embed

class Cartoonset100kDataset(Dataset):
    def __init__(self, attr_txt, root_dir, transform=None):
        self.cartoon_info = []
        with open(attr_txt, "r") as f:
            for c, line in enumerate(f):
                if c != 0:
                    self.cartoon_info.append(line.split())

        self.cartoon_info = pd.DataFrame(self.cartoon_info[1:], columns=["file_name"]+self.cartoon_info[0])

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cartoon_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.cartoon_info.iloc[idx, 0])
        image = io.imread(img_name)
        feature = self.cartoon_info.iloc[idx, 1:].values.astype('float')
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'feature': feature}

        return sample
