import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class FakedditDataset(Dataset):
    """The Fakeddit dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_frame = pd.read_csv(csv_file, delimiter='\t')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.csv_frame.loc[idx, 'id'] + '.jpg')
        image = io.imread(img_name)
        label = self.csv_frame.loc[idx, '2_way_label']
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    print(os.getcwd())
    fake_data = FakedditDataset(csv_file='../../multimodal_validate.tsv', root_dir='../../public_image_set/')
