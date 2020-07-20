import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2, padding = (2, 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 4, padding = (2, 2))
        self.fc1 = nn.Linear(16 * 14 * 14, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
#        print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
#        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
#        print(x.size())
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


