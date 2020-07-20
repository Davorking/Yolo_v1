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

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection(root = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year = '2007', image_set = 'train',
                                             download = True, transform = transform)

trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = 5)

net = LeNet()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(2):
    running_loss = 0.0
    images = trainsetloader[0]
    annotations = trainsetloader[1]
    for i in range(len(images)):
        inputs, labels = images[i], annotations[i]

#        print(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss_value = loss.yolo_loss(outputs, labels)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')

PATH = '.\\voc_net.pth'
torch.save(net.state_dict(), PATH)

#for i, data in enumerate(trainsetloader):
#    images, labels = data[0], data[1]
#    print(images.size())
#    print(len(labels))

#    print(images[0].size())
#    print(len(label[0]))
 #   print(label[0])

#    print(len(label))
#Simulated ground-truth label with batch_size = 2
#sim_label = [[[2, 3, 0.1, 0.2, 0.3, 0.4, 0.5], [3, 3, 0.2, 0.3, 0.4, 0.5, 0.6]], [[3, 4, 0.1, 0.3, 0.4, 0.5, 0.7]]]
#Simulated predicted label with batch_size = 2
#sim_output = torch.rand(2, 7 * 7 * 30)

#loss_value = loss.yolo_loss(sim_output, sim_label)
#print(type(loss_value))
#print('loss: {}'.format(loss_value))

