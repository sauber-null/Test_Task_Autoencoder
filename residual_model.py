import torch
import torch.nn as nn


class Residual_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1), 
            nn.BatchNorm2d(8), 
            nn.ReLU()
        )
        self.down_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.down_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.my_equal = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1 ,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up_block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ReLU()
        )
        self.up_block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 1, 1),
            nn.ReLU()
        )
        self.up_block4 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.my_equal(x3)
        x = self.up_block1(x4)
        x3 = torch.cat([x, x3], dim=1)
        x = self.up_block2(x3)
        x2 = torch.cat([x, x2], dim=1)
        x = self.up_block3(x2)
        x1 = torch.cat([x, x1], dim=1)
        x = self.up_block4(x1)
        return x


def get_residual():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Residual_net().to(device)
    return net
