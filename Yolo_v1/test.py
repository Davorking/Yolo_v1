import random
import torch
import torchvision
import numpy as np
from Yolo import DarkNet
from Yolo import loss
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.optim as optim
from PIL import Image
from PIL import ImageOps

def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()



def VOC_dataloader(path_lines, b_size, pos, jitter):
    t_l_Img = []
    t_l_label = []
#Iterate over the batch of images or quit if EOF reached
    for i in range(b_size): 
        if (pos + i) == len(path_lines):
            break;
        #Read in the Image and Label paths
        t_img_path = lines[pos + i].strip('\n')
        t_label_path = t_img_path
        t_label_path = t_label_path.replace('images', 'labels')
        t_label_path = t_label_path.replace('.jpg', '.txt')
        t_label_path = t_label_path.replace('JPEGImages', 'labels')
#        print('Line {}: {}'.format((i+1), t_img_path))
#        print('Line {}: {}'.format((i+1), t_label_path))

        #Processing the Image
        t_Img = Image.open(t_img_path).convert("RGB")
        t_wid, t_ht = t_Img.size
        dw = jitter * t_wid
        dh = jitter * t_ht

    #    t_l_w = 0
    #    t_r_w = 0
    #    t_u_h = 0
    #    t_l_h = 0

        t_l_w = random.uniform(-dw, dw)
        t_r_w = random.uniform(-dw, dw)
        t_u_h = random.uniform(-dh, dh)
        t_l_h = random.uniform(-dh, dh)

        x_l = t_l_w
        x_r = t_wid + t_r_w
        y_u = t_u_h
        y_l = t_ht + t_l_h

        swid = x_r - x_l
        sht = y_l - y_u

        t_Img = TF.crop(t_Img, y_u, x_l, sht, swid)

        if random.randint(0, 1):
            t_Img = TF.adjust_brightness(t_Img, 1.5)
#            print('Adjust brightness')
        if random.randint(0, 1):
            t_Img = TF.adjust_saturation(t_Img, 1.5)
#            print('Adjust Saturation')

        t_Img = TF.resize(t_Img, (448, 448))
        t_l_Img.append(TF.to_tensor(t_Img))

        #Processing the label
        t_file_2 = open(t_label_path, 'r')
        t_label_line = t_file_2.readlines()
        t_l_obj_label = []
        #Iterate over the lines in the label file
        for j in range(len(t_label_line)):
            t_label_line[j] = t_label_line[j].strip('\n').split()
            t_obj_label = []
            #Convert the label to float
            for k in range(len(t_label_line[j])):
                t_obj_label.append(float(t_label_line[j][k]))
            #print(t_obj_label)

            #Correct the label location given the previous transformation
            t_label_index = t_obj_label[0]
            t_s_x_c = t_obj_label[1]* t_wid / swid - (x_l / swid)
            t_s_y_c = t_obj_label[2]* t_ht / sht - (y_u / sht)
            t_s_wid = t_obj_label[3] * t_wid / swid
            t_s_ht = t_obj_label[4] * t_ht / sht

            #Debug, calculate the current bounding box location
            t_x_min = int(t_s_x_c * 448 - 0.5 * t_s_wid * 448)
            t_y_min = int(t_s_y_c * 448 - 0.5 * t_s_ht * 448)
            t_x_max = int(t_s_x_c * 448 + 0.5 * t_s_wid * 448)
            t_y_max = int(t_s_y_c * 448 + 0.5 * t_s_ht * 448)

            #In case of bounding box goes outside the image
            t_x_min = 0 if t_x_min < 0 else t_x_min
            t_y_min = 0 if t_y_min < 0 else t_y_min
            t_x_max = 447 if t_x_max > 447 else t_x_max
            t_y_max = 447 if t_y_max > 447 else t_y_max
            #Same
            t_x_min = 447 if t_x_min > 447 else t_x_min
            t_y_min = 447 if t_y_min > 447 else t_y_min
            t_x_max = 0 if t_x_max < 0 else t_x_max
            t_y_max = 0 if t_y_max < 0 else t_y_max

            #Calculate new box parameters
            t_x_c = (t_x_min + t_x_max) / 2
            t_y_c = (t_y_min + t_y_max) / 2
            t_wid_bb = t_x_max - t_x_min
            t_height_bb = t_y_max - t_y_min

#            print('Original bounding box width: {}'.format(t_wid_bb))
#            print('Original bounding box height: {}'.format(t_height_bb))

            # Calculate the Normalied size of the bounding-box and the center location of the bounding-box
            t_width = t_height = 448
            t_wid_bb_s = t_wid_bb / t_width
            t_height_bb_s = t_height_bb / t_height
            t_x_s = t_x_c / t_width
            t_y_s = t_y_c / t_height

            # Calculate the corresponding grid square
            t_grid_r = int(t_y_s * 7)
            t_grid_c = int(t_x_s * 7)

            # Calculate the normalized center location w.r.t the grid square
            t_y_c_grid_s = t_y_s * 7 - t_grid_r
            t_x_c_grid_s = t_x_s * 7 - t_grid_c

            t_obj_label = [t_grid_r, t_grid_c, t_x_c_grid_s, t_y_c_grid_s, t_wid_bb_s, t_height_bb_s, t_label_index]

    #         Draw the bounding box in white
#            t_l_Img[-1][0:3, t_y_min, t_x_min:t_x_max] = 1
#            t_l_Img[-1][0:3, t_y_max, t_x_min:(t_x_max + 1)] = 1
#            t_l_Img[-1][0:3, t_y_min:t_y_max, t_x_min] = 1
#            t_l_Img[-1][0:3, t_y_min:t_y_max, t_x_max] = 1

            #Update the corrected label
#            t_obj_label[1] = t_s_x_c
#            t_obj_label[2] = t_s_y_c
#            t_obj_label[3] = t_s_wid
#            t_obj_label[4] = t_s_ht
            t_obj_label = torch.tensor(t_obj_label, dtype = torch.float32)
            t_l_obj_label.append(t_obj_label)

        t_l_label.append(t_l_obj_label)

    output_tensor = torch.stack([t_l_Img[j] for j in range(len(t_l_Img))], dim = 0) 
#    imshow(torchvision.utils.make_grid(output_tensor))
    output_label = t_l_label
    for i in range(len(output_label)):
        print(output_label[i])
    return (pos + b_size), output_tensor, output_label









b_size = 2
train_images = "C:\\Users\\ASUS\\Desktop\\temp_data\\VOC_label\\train.txt"
file_1 = open(train_images, 'r')
lines = file_1.readlines()
random.shuffle(lines)

net = DarkNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0005)
for epoch in range(15):
    running_loss = 0.0

    pos = 0
    for i in range(int(len(lines)/ b_size + 1)):
        b_labels = []
        pos, inputs, t_b_labels = VOC_dataloader(lines, b_size, pos, 0.2)
        print(pos)
        imshow(torchvision.utils.make_grid(inputs))
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


#t_b_image, t_b_labels = VOC_dataloader(lines, 4, 16549, 0.2)
#imshow(torchvision.utils.make_grid(t_b_image))
#for i in range(len(t_b_labels)):
#    print(t_b_labels[i])

