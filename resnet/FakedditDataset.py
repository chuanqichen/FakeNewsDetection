import os
import torch
# from skimage import io, transform
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



class FakedditImageDataset(Dataset):
    """The Fakeddit image dataset class"""

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
        img_name = self.csv_frame.loc[idx, 'id'] + '.jpg'
        img_path = os.path.join(self.root_dir, img_name)
        # image = io.imread(img_path)
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                #print(f"Image {img_name} is {image.mode}!")
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, '2_way_label']
            # sample = {'image': image, 'name': img_name, 'label': label}
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            #print(f"Corrupted image {img_name}")
            return None



def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


if __name__ == "__main__":
    fake_data = FakedditImageDataset(csv_file='/home/akahs/Data/multimodal_test_public.tsv', root_dir='/home/akahs/Data/public_image_set/')
    for k in range(10):
        one_img = fake_data[k]
        print(one_img[0].size, one_img[0].mode)
