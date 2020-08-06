import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

train_images = "C:\\Users\\ASUS\\Desktop\\temp_data\\VOC_label\\train.txt"

file_1 = open(train_images, 'r')
lines = file_1.readlines()

random.shuffle(lines)

t_img_path = lines[0].strip('\n')
print('Line_1: {}'.format(t_img_path))
t_label_path = t_img_path
t_label_path = t_label_path.replace('images', 'labels')
t_label_path = t_label_path.replace('.jpg', '.txt')
t_label_path = t_label_path.replace('JPEGImages', 'labels')
print('Line_1: {}'.format(t_label_path))

Img = Image.open(t_img_path).convert("RGB")

Img = TF.to_tensor(Img)
print(type(Img))

imshow(torchvision.utils.make_grid(Img))



