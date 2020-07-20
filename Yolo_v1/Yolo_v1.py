import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Yolo import loss
from Yolo import DarkNet
from Yolo import LeNet
from Yolo import dataloader
import matplotlib.pyplot as plt
import numpy as np
import math

#The path and parameters needed to be specified depending on the device
#where to download the dataset
data_download_path = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\'
#where to save the model
PATH = '.\\voc_net.pth'
#batch size
b_size = 16

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection(root = data_download_path, year = '2007', image_set = 'train',
                                             download = True, transform = transform)

trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = b_size)

net = DarkNet()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(2):
    running_loss = 0.0
    images = trainsetloader[0]
    annotations = trainsetloader[1]

    for i in range(len(images)):
        inputs, labels = images[i], annotations[i]

        print('\nBatch No: {}'.format(i+1))
#        print(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss_value = loss.yolo_loss(outputs, labels)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
#        print('running loss: {}'.format(loss_value.item()))

        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), PATH)
