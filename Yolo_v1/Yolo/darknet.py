import torch.nn as nn
import torch.nn.functional as F

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding = (3, 3))
        self.conv2 = nn.Conv2d(64, 192, 3, padding = (1, 1))
        self.conv3 = nn.Conv2d(192, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = (1, 1))
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv9 = nn.Conv2d(512, 256, 1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv11 = nn.Conv2d(512, 256, 1)
        self.conv12 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv13 = nn.Conv2d(512, 256, 1)
        self.conv14 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv15 = nn.Conv2d(512, 512, 1)
        self.conv16 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.conv17 = nn.Conv2d(1024, 512, 1)
        self.conv18 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.conv19 = nn.Conv2d(1024, 512, 1)
        self.conv20 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.conv21 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.conv22 = nn.Conv2d(1024, 1024, 3, 2, padding = (1, 1))
        self.conv23 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.conv24 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)
        self.pool = nn.MaxPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.lrelu(self.conv1(x)))
        x = self.pool(self.lrelu(self.conv2(x)))
        x = self.pool(self.lrelu(self.conv6(self.conv5(self.conv4(self.conv(3))))))

        x = self.conv8(self.conv7(x))
        x = self.conv10(self.conv9(x))
        x = self.conv12(self.conv11(x))
        x = self.conv14(self.conv13(x))
        x = self.pool(self.lrelu( self.conv16(self.conv15(x))))

        x = self.conv18(conv17(x))
        x = self.conv20(conv19(x))
        x = self.lrelu(self.conv22(conv21(x)))

        x = self.lrelu(self.conv24(conv23(x)))

        x = x.view(-1, 7 * 7 * 1024)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

