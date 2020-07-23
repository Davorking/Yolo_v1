import sys
import struct
import torch
from Yolo import DarkNet
from Yolo import LeNet
from Yolo import GoogLeNet

#class network:
#    def __init__(self):
#        self.n = 0
#        self.batch = 0
#        self.seen = 0


def load_weights_upto(net, filename):
    s_int = 4
    s_float = 4

    print('Loading weights from {}'.format(filename))

    fp = open(filename, "rb")
    if fp == False: 
        print('Cannot open the file')

    major = struct.unpack('i', fp.read(s_int))[0]
    minor = struct.unpack('i', fp.read(s_int))[0]
    revision = struct.unpack('i', fp.read(s_int))[0]
    
#    iseen = fp.read(4)
#    iseen = struct.unpack('i', fp.read(s_int))[0]

    if ((major*10 + minor) >= 2 and major < 1000 and minor < 1000):
        iseen = struct.unpack('i', fp.read(s_int))[0]
        #fread(net->seen, sizeof(size_t), 1, fp)
    else:
        iseen = 0;
        iseen = struct.unpack('i', fp.read(s_int))[0]
#        fread(&iseen, sizeof(int), 1, fp);
#        *net->seen = iseen;
    print('Major: {}\nMinor: {}\nRevision: {}\niseen: {}\n'.format(major, minor, revision, iseen))

    transpose = True if ((major > 1000) or (minor > 1000)) else False

    test_dict = net.state_dict()
    
#    print(net.state_dict())

    for key in test_dict:
        t_l_weight = []

        if key[0:2] == 'co' and key[-4:] == 'bias':
            print('Processing convolutional layer\'s bias: ')
            t_c_b = test_dict[key].shape[0]
            print('Bias Size: {}'.format(t_c_b))
            for j in range(t_c_b):
                t_w = struct.unpack('f', fp.read(s_float))[0]
                t_l_weight.append(t_w)

            print(test_dict[key])
            print(t_l_weight)

            test_dict[key] = torch.tensor(t_l_weight, dtype = torch.float32)
            print('Number of Bias loaded: {}'.format(test_dict[key].size()))

            print(test_dict[key])

        if key[0:2] == 'co' and key[-4:] == 'ight':
            print('Processing convolutional layer\'s weights: ')
            t_c_out = test_dict[key].shape[0]
            t_c_in = test_dict[key].shape[1]
            t_c_k = test_dict[key].shape[2]

            print('Output Channel: {}\nInput Channel: {}\nKernel Size: {}\n'.format(t_c_out, t_c_in, t_c_k))

        if key[0:2] == 'fc' and key[-4:] == 'bias':
            print('Processing fully connected layer\'s bias: ')
            t_f_b = test_dict[key].shape[0]
            print('Bias Size: {}\n'.format(t_f_b))


        if key[0:2] == 'fc' and key[-4:] == 'ight':
            print('Processing fully connected layer\'s weights: ')
            t_f_out = test_dict[key].shape[0]
            t_f_in = test_dict[key].shape[1]
            print('Output Size: {}\nInput Size: {}\n'.format(t_f_out, t_f_in))

    print('Done!\n');
    fp.close()


filename = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\extraction.conv.weights'

#net = network()
net = GoogLeNet()
net_2 = LeNet()

load_weights_upto(net, filename)

#print(net_2.parameters)
params = list(net_2.parameters())
print('The number of parameters in DarkNet is: {}'.format(len(params)))

#print(net_2.state_dict())
test_dict = net_2.state_dict()

#print(test_dict['conv1.weight'])
#print(test_dict['conv1.weight'].size())
#print(test_dict['conv1.weight'].shape[0])
#print(len(test_dict))

for key in test_dict:
#    print(key[0:2])
#    print(key[-4:])
    if key[0:2] == 'co' and key[-4:] == 'ight':
        print('Processing convolutional layer\'s weights: ')
        t_c_out = test_dict[key].shape[0]
        t_c_in = test_dict[key].shape[1]
        t_c_k = test_dict[key].shape[2]
        print('Output Channel: {}\nInput Channel: {}\nKernel Size: {}\n'.format(t_c_out, t_c_in, t_c_k))

    if key[0:2] == 'co' and key[-4:] == 'bias':
        print('Processing convolutional layer\'s bias: ')
        t_c_b = test_dict[key].shape[0]
        print('Bias Size: {}\n'.format(t_c_b))

    if key[0:2] == 'fc' and key[-4:] == 'ight':
        print('Processing fully connected layer\'s weights: ')
        t_f_out = test_dict[key].shape[0]
        t_f_in = test_dict[key].shape[1]
        print('Output Size: {}\nInput Size: {}\n'.format(t_f_out, t_f_in))

    if key[0:2] == 'fc' and key[-4:] == 'bias':
        print('Processing fully connected layer\'s bias: ')
        t_f_b = test_dict[key].shape[0]
        print('Bias Size: {}\n'.format(t_f_b))





