def yolo_loss(input, label):
    output = label - input
    return output

def IOU(o_l_bb, p_l_bb):
    x_min = o_l_bb[0]
    x_max = o_l_bb[1]
    y_min = o_l_bb[2]
    y_max = o_l_bb[3]

    x_min_p = p_l_bb[0]
    x_max_p = p_l_bb[1]
    y_min_p = p_l_bb[2]
    y_max_p = p_l_bb[3]

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

