import torch
import torchvision
import torchvision.transforms as transforms
from Yolo import loss
from Yolo import DarkNet
from Yolo import dataloader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection(root = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year = '2007', image_set = 'train',
                                             download = True, transform = transform)

trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = 5)


a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

print(a.shape)
print(a)
print(a[2])

b = a.view(2, 2, 2)
print(b)

sim_label = [[[2, 3, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[1, 1, 0.1, 0.3, 0.4, 0.5, 0.7]]]
sim_output = torch.rand(2, 7 * 7 * 30)

temp = sim_output.view(-1, 7, 7, 30)
print(temp.size())
for i in range(temp.shape[0]):
    for j in sim_label[i]:
        t_output = temp[i][j[0]][j[1]]


print(sim_output.shape)


