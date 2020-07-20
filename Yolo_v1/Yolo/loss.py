import torch
import math

def yolo_loss(output, label):
    temp = output.view(-1, 7, 7, 30)
    print(temp.size())
    print(len(label))
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
        for obj in label[i]:
#            show_img = torch.zeros([3, 448, 448])

            #Locate the Responsible Grid from the output label
#            print(type(obj[0]))
#            print(obj[0])
            g_r = int(obj[0])
            g_c = int(obj[1])
            p_bb_data = temp[i][g_r][g_c]

            #Parse the ground-truth label
            o_l_bb = bb_label_parser(obj[0], obj[1], obj[2], obj[3], obj[4], obj[5])

            #Parse the predicted bounding-box No.1
            p_l_bb_1 = bb_label_parser(obj[0], obj[1], p_bb_data[0], p_bb_data[1], p_bb_data[2], p_bb_data[3])
            #Parse the predicted bounding-box No.2
            p_l_bb_2 = bb_label_parser(obj[0], obj[1], p_bb_data[5], p_bb_data[6], p_bb_data[7], p_bb_data[8])

            #Calculate two IoU with Ground-truth
            t_iou_1 = IOU(o_l_bb, p_l_bb_1)
            t_iou_2 = IOU(o_l_bb, p_l_bb_2)

#            print('o_l_bb: {}'.format(o_l_bb))
#            print('p_l_bb_1: {}'.format(p_l_bb_1))
#            print('p_l_bb_2: {}'.format(p_l_bb_2))

#            print('t_iou_1 = {}, t_iou_2 = {}'.format(t_iou_1, t_iou_2))
#            print('p_bb_data: {}'.format(p_bb_data))

            if t_iou_1 > t_iou_2:
                bb_responsible_c = t_iou_1
                p_l_bb_responsible = [p_bb_data[0], p_bb_data[1], p_bb_data[2], p_bb_data[3], p_bb_data[4]]
            else:
                bb_responsible_c = t_iou_2
                p_l_bb_responsible = [p_bb_data[5], p_bb_data[6], p_bb_data[7], p_bb_data[8], p_bb_data[9]]


            loss_1 += lambda_coord * (torch.pow((obj[2] - p_l_bb_responsible[0]), 2) + torch.pow((obj[3] - p_l_bb_responsible[1]), 2))
#            print(type(obj))

#            loss_2 += lambda_coord * ( pow((torch.sqrt(obj_4)-torch.sqrt(p_l_bb_responsible_2)), 2) + 
#                                      pow((torch.sqrt(obj_5)-torch.sqrt(p_l_bb_responsible_3)), 2) )


            loss_2 += lambda_coord * ( torch.pow((torch.sqrt(obj[4])-torch.sqrt(p_l_bb_responsible[2])), 2) + 
                                      torch.pow((torch.sqrt(obj[5])-torch.sqrt(p_l_bb_responsible[3])), 2) )

#            print('torch.sqrt obj[4]: {}'.format(torch.sqrt(obj[4])))
#            print('torch.sqrt obj[5]: {}'.format(torch.sqrt(obj[5])))
#            print('torch.sqrt responsible[2]: {}'.format(torch.sqrt(p_l_bb_responsible[2])))
#            print('torch.sqrt responsible[3]: {}'.format(torch.sqrt(p_l_bb_responsible[3])))


            loss_3 += torch.pow((bb_responsible_c - p_l_bb_responsible[4]), 2)

            for j in range(0, int(obj[6])):
                loss_5 += torch.pow((p_bb_data[10 + j] - 0), 2)
            for j in range(int(obj[6])+1, 20):
                loss_5 += torch.pow((p_bb_data[10 + j] - 0), 2)
            loss_5 += torch.pow((p_bb_data[10 + int(obj[6])] - 1), 2)

        for n_r in range (0, 7):
            for n_c in range (0, 7):
                flag_no_obj = True

                for obj in label[i]:
                    if n_r == obj[0] and n_c == obj[1]:
                        flag_no_obj = False

                if flag_no_obj:
                    p_bb_data = temp[i][n_r][n_c]
                    loss_4 += pow((p_bb_data[4] - 0), 2)
                    loss_4 += pow((p_bb_data[9] - 0), 2)

#        print('loss_1: {}'.format(loss_1))
#        print('loss_2: {}'.format(loss_2))
#        print('loss_3: {}'.format(loss_3))
#        print('loss_4: {}'.format(loss_4))
#        print('loss_5: {}'.format(loss_5))
        loss_whole += loss_1 + loss_2 + loss_3 + loss_4 + loss_5

        loss_whole.clone().detach().requires_grad_(True)
#        print('loss_whole: {}'.format(loss_whole))

    return loss_whole


def IOU(o_l_bb, p_l_bb):
    x_min = o_l_bb[0]
    x_max = o_l_bb[1]
    y_min = o_l_bb[2]
    y_max = o_l_bb[3]

    x_min_p = p_l_bb[0]
    x_max_p = p_l_bb[1]
    y_min_p = p_l_bb[2]
    y_max_p = p_l_bb[3]

    if x_min_p > x_max_p or y_min_p > y_max_p or x_min > x_max or y_min > y_max:
        v_iou = 0
        return v_iou

    if x_min_p > x_max or x_max_p < x_min or y_min_p > y_max or y_max_p < y_min:
        v_iou = 0
    else:
        t_i_x_max = x_max if x_max_p > x_max else x_max_p
        t_i_x_min = x_min if x_min_p < x_min else x_min_p
        t_i_y_max = y_max if y_max_p > y_max else y_max_p
        t_i_y_min = y_min if y_min_p < y_min else y_min_p

        inter_area = (t_i_x_max - t_i_x_min) * (t_i_y_max - t_i_y_min)
        original_area = (x_max - x_min) * (y_max - y_min)
        predicted_area = (x_max_p - x_min_p) * (y_max_p - y_min_p)

        v_iou = inter_area / (original_area + predicted_area - inter_area)

    return v_iou

        
def bb_label_parser(g_r, g_c, g_x_c, g_y_c, width, height):
    t_x_c = (g_c + g_x_c) / 7 * 448.
    t_y_c = (g_r + g_y_c) / 7 * 448.
    t_width = width * 448.
    t_height = height * 448.

    x_min = int(t_x_c - 0.5 * t_width)
    x_max = int(t_x_c + 0.5 * t_width - 1)
    y_min = int(t_y_c - 0.5 * t_height)
    y_max = int(t_y_c + 0.5 * t_height - 1)

    l_bb = [x_min, x_max, y_min, y_max]

    return l_bb
    

def bb_refine4drawing(l_bb):
    t_x_min = l_bb[0] if l_bb[0] > 0 else 0
    t_x_max = l_bb[1] if l_bb[1] < 448 else 448
    t_y_min = l_bb[2] if l_bb[2] > 0 else 0
    t_y_max = l_bb[3] if l_bb[3] < 448 else 448

    out_l_bb = [t_x_min, t_x_max, t_y_min, t_y_max]

    return out_l_bb

