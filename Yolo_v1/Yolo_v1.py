import torch
import torchvision
import torchvision.transforms as transforms
from Yolo import loss
from Yolo import DarkNet
from Yolo import dataloader
import matplotlib.pyplot as plt
import numpy as np
import math

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.VOCDetection(root = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year = '2007', image_set = 'train',
#                                             download = True, transform = transform)

#trainsetloader = dataloader.VOC_DataLoader(trainset, batch_size = 5)

#data, label = trainsetloader

#print(len(data))
#print(len(label))

#print(data[0].size())
#print(len(label[0]))
#print(label[0])






#a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

#print(a.shape)
#print(a)
#print(a[2])

#b = a.view(2, 2, 2)
#print(b)








#Simulated ground-truth label with batch_size = 2
sim_label = [[[2, 3, 0.1, 0.2, 0.3, 0.4, 0.5], [3, 3, 0.2, 0.3, 0.4, 0.5, 0.6]], [[3, 4, 0.1, 0.3, 0.4, 0.5, 0.7]]]
#Simulated predicted label with batch_size = 2
sim_output = torch.rand(2, 7 * 7 * 30)






temp = sim_output.view(-1, 7, 7, 30)
print(temp.size())
#define the parameters
lambda_coord = 5
lambda_noobj = 0.5
#Iterate over the batch
loss_whole = 0
for i in range(temp.shape[0]):
    #iterate over the objectes in the Image
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    loss_4 = 0
    loss_5 = 0
    for obj in sim_label[i]:
        show_img = torch.zeros([3, 448, 448])

        #Locate the Responsible Grid from the output label
        p_bb_data = temp[i][obj[0]][obj[1]]

        #Parse the ground-truth label
        o_l_bb = loss.bb_label_parser(obj[0], obj[1], obj[2], obj[3], obj[4], obj[5])

        #Parse the predicted bounding-box No.1
#        p_l_bb_1 = loss.bb_label_parser(obj[0], obj[1], p_bb_data[0], p_bb_data[1], p_bb_data[2], p_bb_data[3])
        p_l_bb_1 = loss.bb_label_parser(obj[0], obj[1], 0.1, 0.2, 0.29, 0.4)
        #Parse the predicted bounding-box No.2
        p_l_bb_2 = loss.bb_label_parser(obj[0], obj[1], p_bb_data[5], p_bb_data[6], p_bb_data[7], p_bb_data[8])

        #Calculate two IoU with Ground-truth
        t_iou_1 = loss.IOU(o_l_bb, p_l_bb_1)
        t_iou_2 = loss.IOU(o_l_bb, p_l_bb_2)
        print('t_iou_1 = {}, t_iou_2 = {}'.format(t_iou_1, t_iou_2))

        if t_iou_1 > t_iou_2:
            bb_responsible_c = t_iou_1
            p_l_bb_responsible = [p_bb_data[0], p_bb_data[1], p_bb_data[2], p_bb_data[3], p_bb_data[4]]
        else:
            bb_responsible_c = t_iou_2
            p_l_bb_responsible = [p_bb_data[5], p_bb_data[6], p_bb_data[7], p_bb_data[8], p_bb_data[9]]


        loss_1 += lambda_coord * (pow((obj[2] - p_l_bb_responsible[0]), 2) + pow((obj[3] - p_l_bb_responsible[1]), 2))
        loss_2 += lambda_coord * ( pow((math.sqrt(obj[4])-math.sqrt(p_l_bb_responsible[2])), 2) + pow((math.sqrt(obj[5])-math.sqrt(p_l_bb_responsible[3])), 2) )
        loss_3 += pow((bb_responsible_c - p_l_bb_responsible[4]), 2)

        for j in range(0, int(obj[6])):
            loss_5 += pow((p_bb_data[10 + j] - 0), 2)
        for j in range(int(obj[6])+1, 20):
            loss_5 += pow((p_bb_data[10 + j] - 0), 2)
        loss_5 += pow((p_bb_data[10 + int(obj[6])] - 1), 2)

    for n_r in range (0, 7):
        for n_c in range (0, 7):
            flag_no_obj = True

            for obj in sim_label[i]:
                if n_r == obj[0] and n_c == obj[1]:
                    flag_no_obj = False

            if flag_no_obj:
                p_bb_data = temp[i][n_r][n_c]
                loss_4 += pow((p_bb_data[4] - 0), 2)
                loss_4 += pow((p_bb_data[9] - 0), 2)


    print('loss_1: {}'.format(loss_1))
    print('loss_2: {}'.format(loss_2))
    print('loss_3: {}'.format(loss_3))
    print('loss_4: {}'.format(loss_4))
    print('loss_5: {}'.format(loss_5))
    loss_whole += loss_1 + loss_2 + loss_3 + loss_4 + loss_5
    print('loss_whole: {}'.format(loss_whole))


        #In case of drawing outside the 448x448 canvas
##        p_d_l_bb_1 = loss.bb_refine4drawing(p_l_bb_1)
#        p_d_l_bb_2 = loss.bb_refine4drawing(p_l_bb_2)
#        o_d_l_bb = loss.bb_refine4drawing(o_l_bb)

        #Mark Ground-truth box area with white
#        show_img[0:3, o_d_l_bb[2]:o_d_l_bb[3], o_d_l_bb[0]:o_d_l_bb[1]] = 1

        #Mark No.1 Predicted box area with yellow
#        show_img[0, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 1
#        show_img[1, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 1
#        show_img[2, p_d_l_bb_1[2]:p_d_l_bb_1[3], p_d_l_bb_1[0]:p_d_l_bb_1[1]] = 0

#        dataloader.imshow(show_img)

#print(sim_output.shape)


