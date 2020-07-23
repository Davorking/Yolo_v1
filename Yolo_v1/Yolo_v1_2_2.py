import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from Yolo import loss
from Yolo import DarkNet
from Yolo import dataloader
from Yolo import weightloader
import matplotlib.pyplot as plt
import numpy as np

#The path and parameters needed to be specified depending on the device
#where to access the dataset
data_dataset_path = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\'
#where to load the model trained from phase 1
PATH = '.\\yolo_darknet-phase_1.pth'

#batch size
b_size = 64

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print('Start loading the data!')
trainset = torchvision.datasets.VOCDetection(root = data_dataset_path, year = '2012', image_set = 'trainval',
                                             download = True, transform = transform)

trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = b_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = DarkNet()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)

print('Start loading the parameter!')
net.load_state_dict(torch.load(PATH))

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0005)
#lr = 0.01 for first 75, 0.001 for next 30, 0.0001 for the last 30 epochs
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [75, 105], 0.1)

print('Start Training (phase_2, total epoch: 135)')

for epoch in range(135):
    running_loss = 0.0
    images = trainsetloader[0]
    annotations = trainsetloader[1]

    for i in range(len(images)):
        inputs, labels = images[i], annotations[i]

#        print('\nBatch No: {}'.format(i+1))
#        print(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss_value = loss.yolo_loss(outputs, labels)
        loss_value.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_value.item()
#        print('running loss: {}'.format(loss_value.item()))

        if i % 10 == 9:
            print('[Epoch: %d, Batch: %5d] Average Loss per Batch: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), PATH)
print('Successfully saved the model!')


