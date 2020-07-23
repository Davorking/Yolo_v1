import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv01 = nn.Conv2d(3, 64, 7, 2, padding = (3, 3))
        self.conv02 = nn.Conv2d(64, 192, 3, padding = (1, 1))
        self.conv03 = nn.Conv2d(192, 128, 1)
        self.conv04 = nn.Conv2d(128, 256, 3, padding = (1, 1))
        self.conv05 = nn.Conv2d(256, 256, 1)
        self.conv06 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv07 = nn.Conv2d(512, 256, 1)
        self.conv08 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.conv09 = nn.Conv2d(512, 256, 1)
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
        self.fc1 = nn.Linear(7 * 7 * 1024, 1000)
        self.pool = nn.MaxPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.lrelu(self.conv01(x)))
        x = self.pool(self.lrelu(self.conv02(x)))
        x = self.pool(self.lrelu(self.conv06(self.conv05(self.conv04(self.conv03(x))))))

        x = self.conv08(self.conv07(x))
        x = self.conv10(self.conv09(x))
        x = self.conv12(self.conv11(x))
        x = self.conv14(self.conv13(x))
        x = self.pool(self.lrelu( self.conv16(self.conv15(x))))

        x = self.lrelu(self.conv18(self.conv17(x)))
        x = self.lrelu(self.conv20(self.conv19(x)))

        x = x.view(-1, 7 * 7 * 1024)

        x = F.relu(self.fc1(x))
        return x

