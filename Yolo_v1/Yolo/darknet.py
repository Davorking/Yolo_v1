import torch.nn as nn
import torch.nn.functional as F

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.conv01 = nn.Conv2d(3, 64, 7, 2, padding = (3, 3))
        self.bn01 = nn.BatchNorm2d(num_features = 64)

        self.conv02 = nn.Conv2d(64, 192, 3, padding = (1, 1))
        self.bn02 = nn.BatchNorm2d(num_features = 192)

        self.conv03 = nn.Conv2d(192, 128, 1)
        self.bn03 = nn.BatchNorm2d(num_features = 128)
        self.conv04 = nn.Conv2d(128, 256, 3, padding = (1, 1))
        self.bn04 = nn.BatchNorm2d(num_features = 256)
        self.conv05 = nn.Conv2d(256, 256, 1)
        self.bn05 = nn.BatchNorm2d(num_features = 256)
        self.conv06 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.bn06 = nn.BatchNorm2d(num_features = 512)
        
        self.conv07 = nn.Conv2d(512, 256, 1)
        self.bn07 = nn.BatchNorm2d(num_features = 256)
        self.conv08 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.bn08 = nn.BatchNorm2d(num_features = 512)
        self.conv09 = nn.Conv2d(512, 256, 1)
        self.bn09 = nn.BatchNorm2d(num_features = 256)
        self.conv10 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.bn10 = nn.BatchNorm2d(num_features = 512)
        self.conv11 = nn.Conv2d(512, 256, 1)
        self.bn11 = nn.BatchNorm2d(num_features = 256)
        self.conv12 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.bn12 = nn.BatchNorm2d(num_features = 512)
        self.conv13 = nn.Conv2d(512, 256, 1)
        self.bn13 = nn.BatchNorm2d(num_features = 256)
        self.conv14 = nn.Conv2d(256, 512, 3, padding = (1, 1))
        self.bn14 = nn.BatchNorm2d(num_features = 512)
        self.conv15 = nn.Conv2d(512, 512, 1)
        self.bn15 = nn.BatchNorm2d(num_features = 512)
        self.conv16 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.bn16 = nn.BatchNorm2d(num_features = 1024)

        self.conv17 = nn.Conv2d(1024, 512, 1)
        self.bn17 = nn.BatchNorm2d(num_features = 512)
        self.conv18 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.bn18 = nn.BatchNorm2d(num_features = 1024)
        self.conv19 = nn.Conv2d(1024, 512, 1)
        self.bn19 = nn.BatchNorm2d(num_features = 512)
        self.conv20 = nn.Conv2d(512, 1024, 3, padding = (1, 1))
        self.bn20 = nn.BatchNorm2d(num_features = 1024)

        self.conv21 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.bn21 = nn.BatchNorm2d(num_features = 1024)
        self.conv22 = nn.Conv2d(1024, 1024, 3, 2, padding = (1, 1))
        self.bn22 = nn.BatchNorm2d(num_features = 1024)
        self.conv23 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.bn23 = nn.BatchNorm2d(num_features = 1024)
        self.conv24 = nn.Conv2d(1024, 1024, 3, padding = (1, 1))
        self.bn24 = nn.BatchNorm2d(num_features = 1024)

        self.fc1 = nn.Linear(7 * 7 * 1024, 4096)
#        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)

        self.pool = nn.MaxPool2d(2, 2)
        self.lrelu = nn.LeakyReLU(0.1)

#    def forward(self, x):

#        x = self.pool(self.lrelu(self.conv01(x)))
#        x = self.pool(self.lrelu(self.conv02(x)))
#        x = self.pool(self.lrelu(self.conv06(self.conv05(self.conv04(self.conv03(x))))))
#
#        x = self.conv08(self.conv07(x))
#        x = self.conv10(self.conv09(x))
#        x = self.conv12(self.conv11(x))
#        x = self.conv14(self.conv13(x))
#        x = self.pool(self.lrelu( self.conv16(self.conv15(x))))
#
#        x = self.conv18(self.conv17(x))
#        x = self.conv20(self.conv19(x))
#        x = self.lrelu(self.conv22(self.conv21(x)))
#
#        x = self.lrelu(self.conv24(self.conv23(x)))
#
#        x = x.view(-1, 7 * 7 * 1024)
#
#        x = F.relu(self.fc1(x))
##        x = self.dropout(F.relu(self.fc1(x)))
#        x = F.relu(self.fc2(x))
#        return x






    def forward(self, x):

        x = self.pool(self.lrelu(self.bn01(self.conv01(x))))
        x = self.pool(self.lrelu(self.bn02(self.conv02(x))))

        x = self.lrelu(self.bn03(self.conv03(x)))
        x = self.lrelu(self.bn04(self.conv04(x)))
        x = self.lrelu(self.bn05(self.conv05(x)))
        x = self.pool(self.lrelu(self.bn06(self.conv06(x))))


        x = self.lrelu(self.bn07(self.conv07(x)))
        x = self.lrelu(self.bn08(self.conv08(x)))
        x = self.lrelu(self.bn09(self.conv09(x)))
        x = self.lrelu(self.bn10(self.conv10(x)))
        x = self.lrelu(self.bn11(self.conv11(x)))
        x = self.lrelu(self.bn12(self.conv12(x)))
        x = self.lrelu(self.bn13(self.conv13(x)))
        x = self.lrelu(self.bn14(self.conv14(x)))
        x = self.lrelu(self.bn15(self.conv15(x)))
        x = self.pool(self.lrelu(self.bn16(self.conv16(x))))


        x = self.lrelu(self.bn17(self.conv17(x)))
        x = self.lrelu(self.bn18(self.conv18(x)))
        x = self.lrelu(self.bn19(self.conv19(x)))
        x = self.lrelu(self.bn20(self.conv20(x)))


        x = self.lrelu(self.bn21(self.conv21(x)))
        x = self.lrelu(self.bn22(self.conv22(x)))
        x = self.lrelu(self.bn23(self.conv23(x)))
        x = self.lrelu(self.bn24(self.conv24(x)))
        x = x.view(-1, 7 * 7 * 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
