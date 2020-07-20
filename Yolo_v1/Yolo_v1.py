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

#trainset = torchvision.datasets.VOCDetection(root = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year = '2007', image_set = 'train',
#                                             download = True, transform = transform)

#trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = 5)

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

print(a.shape)
print(a)
print(a[2])

b = a.view(2, 2, 2)
print(b)

#Simulated ground-truth label with batch_size = 2
sim_label = [[[2, 3, 0.1, 0.2, 0.3, 0.4, 0.5], [3, 3, 0.2, 0.3, 0.4, 0.5, 0.6]], [[3, 4, 0.1, 0.3, 0.4, 0.5, 0.7]]]
#Simulated predicted label with batch_size = 2
sim_output = torch.rand(2, 7 * 7 * 30)


temp = sim_output.view(-1, 7, 7, 30)
print(temp.size())
#Iterate over the batch
for i in range(temp.shape[0]):
    #iterate over the objectes in the Image
    for obj in sim_label[i]:
        show_img = torch.zeros([3, 448, 448])

        #Locate the Responsible Grid
        p_bb_data = temp[i][obj[0]][obj[1]]

        #Parse the ground-truth label
        o_l_bb = loss.bb_label_parser(obj[0], obj[1], obj[2], obj[3], obj[4], obj[5])

        #Parse the predicted label No.1
#        p_l_bb_1 = loss.bb_label_parser(obj[0], obj[1], p_bb_data[0], p_bb_data[1], p_bb_data[2], p_bb_data[3])
        p_l_bb_1 = loss.bb_label_parser(obj[0], obj[1], 0.1, 0.2, 0.29, 0.4)
        #Parse the predicted label No.2
        p_l_bb_2 = loss.bb_label_parser(obj[0], obj[1], p_bb_data[5], p_bb_data[6], p_bb_data[7], p_bb_data[8])

        #Calculate two IoU with Ground-truth
        t_iou_1 = loss.IOU(o_l_bb, p_l_bb_1)
        t_iou_2 = loss.IOU(o_l_bb, p_l_bb_2)
        print('t_iou_1 = {}, t_iou_2 = {}'.format(t_iou_1, t_iou_2))

        #In case of drawing outside the 448x448 canvas
        p_d_l_bb_1 = loss.bb_refine4drawing(p_l_bb_1)
        p_d_l_bb_2 = loss.bb_refine4drawing(p_l_bb_2)
        o_d_l_bb = loss.bb_refine4drawing(o_l_bb)

        #Mark Ground-truth box area with white
        show_img[0:3, o_d_l_bb[2]:o_d_l_bb[3], o_d_l_bb[0]:o_d_l_bb[1]] = 1

        #Mark No.1 Predicted box area with yellow
        show_img[0, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 1
        show_img[1, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 1
        show_img[2, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 0

        dataloader.imshow(show_img)

print(sim_output.shape)


