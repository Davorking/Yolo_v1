import struct
import torch
from Yolo import DarkNet
from Yolo import LeNet
from Yolo import GoogLeNet

#Since the weight file in Yolo official website is customized binary file, build the python version IO

#Aims at building the python version of load_weight_upto in github/pjreddie/darknet/src/parse.c (line 1218)
def load_weights_upto(net, filename):
    #The size of data type in C 
    s_int = 4
    s_float = 4

    print('Loading weights from {}'.format(filename))

    fp = open(filename, "rb")
    if fp == False: 
        print('Cannot open the file')

    #Imitate the process in Original function
    #'i' stands for data type int, the way to load binary file in python
    major = struct.unpack('i', fp.read(s_int))[0]
    minor = struct.unpack('i', fp.read(s_int))[0]
    revision = struct.unpack('i', fp.read(s_int))[0]
    
    if ((major*10 + minor) >= 2 and major < 1000 and minor < 1000):
        iseen = struct.unpack('i', fp.read(s_int))[0]
    else:
        iseen = 0;
        iseen = struct.unpack('i', fp.read(s_int))[0]
    transpose = True if ((major > 1000) or (minor > 1000)) else False
    #show the result
    print('Major: {}\nMinor: {}\nRevision: {}\niseen: {}\nTranspose: {}'.format(major, minor, revision, iseen, transpose))

    #Load the Network parameters to modify (pytorch)
    #Since the original code write the bias terms first, and model.state_dict() is opposite, change the loading order
    test_dict = net.state_dict()
    t_l_bias = []
    t_l_weight = []

    #iterate over the state_dict()
    for key in test_dict:
        #if Convolutional layer
        if key[0:2] == 'co': 
            #the key start with 'co', end with 'ight', eg. conv1.weight.
            if key[-4:] == 'ight':
                print('\nProcessing convolutional layer\'s weights: ')
                t_c_out = test_dict[key].shape[0]
                t_c_in = test_dict[key].shape[1]
                t_c_k = test_dict[key].shape[2]
                print('Output Channel: {}\nInput Channel: {}\nKernel Size: {}'.format(t_c_out, t_c_in, t_c_k))
                #Number of parameters in bias
                t_c_b = t_c_out
                #Number of parameters in weights
                t_num_w = t_c_out * (t_c_in * t_c_k * t_c_k)
                #Read in the Bias term of the layer
                for j in range(t_c_b):
                    t_w = struct.unpack('f', fp.read(s_float))[0]
                    t_l_bias.append(t_w)
                #Read in the weight term of the layer
                for j in range(t_num_w):
                    t_w = struct.unpack('f', fp.read(s_float))[0]
                    t_l_weight.append(t_w)
                #Specify the data type to avoid memory leaking
                t_t_weight = torch.tensor(t_l_weight, dtype = torch.float32)
                t_t_weight = t_t_weight.view(t_c_out, t_c_in, t_c_k, t_c_k)
                test_dict[key] = t_t_weight
                t_l_weight = []
                print('Size of Weight loaded: {}'.format(test_dict[key].size()))
            #the key start with 'co', end with 'bias', eg. conv1.bias.
            if key[-4:] == 'bias':
                print('Processing convolutional layer\'s bias: ')
                print('Bias Size: {}'.format(t_c_b))
                test_dict[key] = torch.tensor(t_l_bias, dtype = torch.float32)
                t_l_bias = []
                print('Size of Bias loaded: {}'.format(test_dict[key].size()))

        #if Fully Connected layer
        if key[0:2] == 'fc': 
            #the key start with 'fc', end with 'ight', eg. fc1.weight.
            if key[-4:] == 'ight':
                print('\nProcessing fully connected layer\'s weights: ')
                t_f_out = test_dict[key].shape[0]
                t_f_in = test_dict[key].shape[1]
                print('Output Size: {}\nInput Size: {}'.format(t_f_out, t_f_in))

                t_num_w = t_f_out * t_f_in
                t_f_b = t_f_out

                for j in range(t_f_b):
                    t_w = struct.unpack('f', fp.read(s_float))[0]
                    t_l_bias.append(t_w)
#                for j in range(t_num_w):
#                    t_w = struct.unpack('f', fp.read(s_float))[0]
#                    t_l_bias.append(t_w)
#                t_t_weight = torch.tensor(t_l_weight, dtype = torch.float32)
#                t_t_weight = t_t_weight.view(t_f_out, t_f_in)
#                test_dict[key] = t_t_weight
#                t_l_weight = []
            #the key start with 'fc', end with 'bias', eg. fc1.bias.
            if key[0:2] == 'fc' and key[-4:] == 'bias':
                print('Processing fully connected layer\'s bias: ')
                print('Bias Size: {}\n'.format(t_f_b))
                test_dict[key] = torch.tensor(t_l_bias, dtype = torch.float32)
                t_l_bias = []
                print('Size of Bias loaded: {}'.format(test_dict[key].size()))

    print('Done!\n');
    fp.close()


filename = 'C:\\Users\\ASUS\\Desktop\\temp_data\\data\\Yolo_v1\\extraction.conv.weights'
net = GoogLeNet()
load_weights_upto(net, filename)


#The code used in trial phase

#net_2 = LeNet()
#print(net_2.parameters)
#params = list(net_2.parameters())
#print('The number of parameters in DarkNet is: {}'.format(len(params)))

#print(net_2.state_dict())
#test_dict = net_2.state_dict()

#print(test_dict['conv1.weight'])
#print(test_dict['conv1.weight'].size())
#print(test_dict['conv1.weight'].shape[0])
#print(len(test_dict))

#for key in test_dict:
#    print(key[0:2])
#    print(key[-4:])
#    if key[0:2] == 'co' and key[-4:] == 'ight':
#        print('Processing convolutional layer\'s weights: ')
#        t_c_out = test_dict[key].shape[0]
#        t_c_in = test_dict[key].shape[1]
#        t_c_k = test_dict[key].shape[2]
#        print('Output Channel: {}\nInput Channel: {}\nKernel Size: {}\n'.format(t_c_out, t_c_in, t_c_k))

#    if key[0:2] == 'co' and key[-4:] == 'bias':
#        print('Processing convolutional layer\'s bias: ')
#        t_c_b = test_dict[key].shape[0]
#        print('Bias Size: {}\n'.format(t_c_b))

#    if key[0:2] == 'fc' and key[-4:] == 'ight':
#        print('Processing fully connected layer\'s weights: ')
#        t_f_out = test_dict[key].shape[0]
#        t_f_in = test_dict[key].shape[1]
#        print('Output Size: {}\nInput Size: {}\n'.format(t_f_out, t_f_in))

#    if key[0:2] == 'fc' and key[-4:] == 'bias':
#        print('Processing fully connected layer\'s bias: ')
#        t_f_b = test_dict[key].shape[0]
#        print('Bias Size: {}\n'.format(t_f_b))





