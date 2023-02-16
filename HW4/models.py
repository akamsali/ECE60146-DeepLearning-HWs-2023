import torch
import torch.nn as nn
import torch.nn.functional as F

# Net 1
class HW4Net1(nn.Module):
    def __init__(self):
        super(HW4Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        # b x 3 x 64 x 64 -> b x 16 x 62 x 62 -> b x 16 x 31 x 31
        x = self.pool(F.relu(self.conv1(x)))
        # b x 16 x 31 x 31 -> b x 32 x 29 x 29 -> b x 32 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        # print("x: ", x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Net 2
class HW4Net2(nn.Module):
    def __init__(self):
        super(HW4Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        # b x 3 x 64 x 64 -> b x 16 x 64 x 64 -> b x 16 x 32 x 32
        x = self.pool(F.relu(self.conv1(x)))
        # b x 16 x 32 x 32 -> b x 32 x 32 x 32 -> b x 32 x 16 x 16
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Net 3
class HW4Net3(nn.Module):
    def __init__(self, num_chained=10) -> None:
        super(HW4Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.conv_chained = nn.ModuleList(
            [nn.Conv2d(32, 32, 3, padding=1) for _ in range(num_chained)]
        )
        # since
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b x 3 x 64 x 64 -> b x 16 x 64 x 64 -> b x 16 x 32 x 32
        x = self.pool(F.relu(self.conv1(x)))
        # b x 16 x 32 x 32 -> b x 32 x 32 x 32 -> b x 32 x 16 x 16
        x = self.pool(F.relu(self.conv2(x)))

        # since no pooling same output b x 32 x 16 x 16 -> b x 32 x 16 x 16
        for cl in self.conv_chained:
            x = cl(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
