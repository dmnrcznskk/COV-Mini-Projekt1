import torch
import torch.nn as nn


class PetsCNN(nn.Module):
    def __init__(
        self,
        num_classes=37,
        use_third_conv=False,
        use_batchnorm=False,
        dropout=None,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)

        self.use_third_conv = use_third_conv
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) if use_third_conv else None
        self.bn3   = nn.BatchNorm2d(128)               if use_third_conv else None

        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else None

        if use_third_conv:
            fc1_in = 128 * 26 * 26
        else:
            fc1_in = 64 * 54 * 54

        self.fc1 = nn.Linear(fc1_in, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        if self.use_third_conv:
            x = self.conv3(x)
            if self.use_batchnorm:
                x = self.bn3(x)
            x = self.relu(x)
            x = self.pool(x)

        x = x.view(x.size(0), -1)

        if self.dropout:
            x = self.dropout(x)

        x = self.relu(self.fc1(x))

        if self.dropout:
            x = self.dropout(x)

        x = self.fc2(x)
        return x
