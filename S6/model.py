import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias=True),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.05)
        )

        self.pool1 = nn.MaxPool2d(2,2) 

        self.pool2 = nn.MaxPool2d(2, 2) 

        self.fc = nn.Sequential(
            nn.Linear(800, 10, bias=True)
        )

        

    def forward(self, x):

        x = self.conv1(x)  # 28 -> 26 -> 24 -> 12 ->
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)