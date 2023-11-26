from torch import nn
import torch
import torch.nn.functional as F

class KannadaMNISTCNN(nn.Module):
    def __init__(self):
        super(KannadaMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, activation='relu')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, activation='relu')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 64) # Adjust input features according to your conv layers output
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10) # Assuming 10 classes for Kannada digits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
