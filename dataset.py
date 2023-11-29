import glob
import os
import cv2

from torch.utils.data import Dataset

import numpy as np

import torch

class DogsCats(Dataset):
    """Dogs vs Cats dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        for i in range(2):
            for image_path in glob.glob(os.path.join(self.root_dir, str(i), "*")):
                self.images.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0
        
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float)
        if self.transform:
            image = self.transform(image)

        return image, label

class DogsCatsTest(Dataset):
    """Dogs vs Cats dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        self.images = []

        for image_path in glob.glob(os.path.join(self.root_dir, "*")):
            self.images.append(image_path)
      

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, image_name.split("/")[-1].split(".")[0]
