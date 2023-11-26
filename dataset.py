import glob
import os
import cv2

from torch.utils.data import Dataset





class KannadaMNIST(Dataset):
    """Kannada MNIST dataset."""

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

        for i in range(10):
            for image_path in glob.glob(os.path.join(self.root_dir, str(i), "*.png")):
                self.images.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
