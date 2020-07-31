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

#The parameter needs to be specified under Linux:
#weightloader l: 49, 54.
#Yolo_v1_1_2 l:  20, 23, 27.

#The path and parameters needed to be specified depending on the device
#where to access the dataset
#data_dataset_path = '~/Laboratory/data'
data_dataset_path = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data'
#where to find the pre-trained on ImageNet weight file
#data_weight_path = '/root/Laboratory/data/extraction.conv.weights'
data_weight_path = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\extraction.conv.weights'
n_conv = 20
#where to save the model
#PATH = './yolo_darknet-phase_1.pth'
PATH = '.\\yolo_darknet-phase_1.pth'

#batch size
b_size = 4

#Define transform for the dataset
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Load the dataset
print('[INFO] Start loading the data!')
trainset = torchvision.datasets.VOCDetection(root = data_dataset_path, year = '2007', image_set = 'train',
                                             download = True, transform = transform)
trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = b_size)

#Construct the Network
net = DarkNet()

#Check if there was multiple GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("[INFO] Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)
net.to(device)

#Loading the Pretrained-In-ImageNet weights provided by the author
print('[INFO] Start loading the convolutional layers\' weights')
weightloader.load_weights_upto(net, data_weight_path)

#Freeze the loaded weights
#for i, param in enumerate(net.parameters(), 0):
#    if i < n_conv * 2:
#        param.requires_grad = False

#Define learning rate, momentum, etc...
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)

#Training goes here
print('Start Training (phase_1, total epoch: 15)')
for epoch in range(15):
    running_loss = 0.0
    images = trainsetloader[0]
    annotations = trainsetloader[1]

    for i in range(len(images)):
        b_labels = []
        inputs, t_b_labels = images[i], annotations[i]
        #load Data into the GPU
        inputs = inputs.to(device)
        for t_i_labels in t_b_labels:
            b_m_labels = []
            for t_i_m_labels in t_i_labels:
                t_i_m_labels = t_i_m_labels.to(device)
                b_m_labels.append(t_i_m_labels)
            b_labels.append(b_m_labels)

#        print('\nBatch No: {}'.format(i+1))
#        print(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss_value = loss.yolo_loss(outputs, b_labels)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
#        print('running loss: {}'.format(loss_value.item()))

        if i % 10 == 9:
            print('[Epoch: %d, Batch: %5d] Average Loss per Batch: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('[INFO] Finished Training!')
torch.save(net.state_dict(), PATH)
print('[INFO] Successfully saved the model!')


##m=nn.BatchNorm2d(2,affine=True)
#input=torch.randn(1,2,3,4)
#output=m(input)
 
#print(input)
#print(m.weight)
#print(m.bias)
#print(output)
#print(output.size())

