import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#import inspect


def imshow(img):
    #    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()])
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.VOCDetection(root='C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year='2007', image_set='train',
                                             download=True, transform=transform)

shuffle = True
b_size = 1

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 
          'bottle', 'bus', 'car', 'cat',
          'chair', 'cow', 'diningtable', 'dog',
          'horse', 'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor']
return_anno = []

for i, data in enumerate(trainset, 0):
    images, annotations = data
    t_anno_o = annotations['annotation']['object']

    print(type(images))
    print(images.size())
    #print('Annotation: ')
    # print(annotations['annotation'])

    #new label
    t_out_annotation = []

    for k in t_anno_o:
        t_anno_bb = k['bndbox']
        t_anno_n = k['name']
        t_label_index = -1

        for t_label in classes:
            if t_anno_n == t_label:
                t_label_index = classes.index(t_label)
                break

        print('\nBounding-box name: {}'.format(t_anno_n))
        print('Bounding-box label index: {}'.format(t_label_index))

        t_anno_size = annotations['annotation']['size']
        # Get the size of the original image
        t_width = int(t_anno_size['width'])
        t_height = int(t_anno_size['height'])

        # Get the size of the bounding-box in the original image
        t_x_min = int(t_anno_bb['xmin'])
        t_x_max = int(t_anno_bb['xmax'])
        t_y_min = int(t_anno_bb['ymin'])
        t_y_max = int(t_anno_bb['ymax'])

        # Calculate the center of the bounding-box in the original image
        t_x_c = (t_x_min + t_x_max) / 2
        t_y_c = (t_y_min + t_y_max) / 2
        t_wid_bb = t_x_max - t_x_min
        t_height_bb = t_y_max - t_y_min

        print('Original bounding box width: {}'.format(t_wid_bb))
        print('Original bounding box height: {}'.format(t_height_bb))

        # Calculate the Normalied size of the bounding-box and the center location of the bounding-box
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

        print('Responsible grid square: (row: {}, col: {})'.format(
            t_grid_r, t_grid_c))
        print('Normalized center location inside the grid: (x_c: {}, y_c: {})'.format(
            t_x_c_grid_s, t_y_c_grid_s))
        print('Normalized bounding-box size: (width: {}, height: {})'.format(t_wid_bb_s, t_height_bb_s))

        # Calculate the bounding-box parameters in the resized image
        t_wid_bb = int(t_wid_bb_s * 448)
        t_height_bb = int(t_height_bb_s * 448)
        t_x_c = int(t_x_s * 448)
        t_y_c = int(t_y_s * 448)

        print('New bounding-box width: {}'.format(t_wid_bb))
        print('New bounding-box height: {}'.format(t_height_bb))
        print('x_c: {}'.format(t_x_c))
        print('y_c: {}'.format(t_y_c))

        # Calculate the bounding-box corner in the new resized image
        t_y_min = int(t_y_c - t_height_bb * 0.5)
        t_y_max = int(t_y_c + t_height_bb * 0.5 - 1)
        t_x_min = int(t_x_c - t_wid_bb * 0.5)
        t_x_max = int(t_x_c + t_wid_bb * 0.5 - 1)

        # Draw the center of the bounding box in red
        images[0, t_y_c, t_x_c] = 1
        images[1, t_y_c, t_x_c] = 0
        images[2, t_y_c, t_x_c] = 0

        # Draw the bounding box in white
        images[0:3, t_y_min, t_x_min:t_x_max] = 1
        images[0:3, t_y_max, t_x_min:(t_x_max + 1)] = 1
        images[0:3, t_y_min:t_y_max, t_x_min] = 1
        images[0:3, t_y_min:t_y_max, t_x_max] = 1

        #Organize the label in the following order
        t_annotation_m = [t_grid_r, t_grid_c, t_x_c_grid_s, t_y_c_grid_s, t_wid_bb_s, t_height_bb_s, t_label_index]
        t_out_annotation.append(t_annotation_m)

#            print(t_anno_bb)
    # Draw grid lines for reference purpose
    for j in range(1, 7):
        images[0, 64 * j - 1, 0:448] = 0
        images[1, 64 * j - 1, 0:448] = 1
        images[2, 64 * j - 1, 0:448] = 0

        images[0, 0:448, 64 * j - 1] = 0
        images[1, 0:448, 64 * j - 1] = 1
        images[2, 0:448, 64 * j - 1] = 0
        print('{} th grid line has been drawn'.format(j))

    #return_anno.append(t_out_annotation)
    print(t_out_annotation)
    #        print(t_anno_o)
    print('Image size: {}'.format(images.size()))
    print('This is the No. {} image'.format(i+1))
    if (i == 115):
        imshow(images)
# print(next(iter(trainset)))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.VOCDetection(root='C:\\Users\\ASUS\\Desktop\\temp_data\\data\\', year='2007', image_set='val',
                                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=b_size,
                                         shuffle=True, num_workers=0)


#dataiter = iter(trainloader)
#images, annotations = dataiter.next()
# print(images.size())

# for i, data in enumerate(trainloader, 0):
#    images, annotations = data
#    t_anno_o = annotations['annotation']['object']
#    if isinstance(t_anno_o, list) and len(t_anno_o) > 1:
#        print(t_anno_o)
#        imshow(torchvision.utils.make_grid(images))

#t_anno_o = annotations['annotation']['object']
#t_anno_n = annotations['annotation']['object']['name']
#t_anno_b = annotations['annotation']['object']['bndbox']

# print(t_anno_o)
# print(t_anno_b)
# print(t_anno_2['bndbox'])

# print(annotations['annotation']['object']['name'])
# print(annotations['annotation']['object']['bndbox'])

# imshow(torchvision.utils.make_grid(images))

