

def yolo_loss(input, label):
    output = label - input
    return output

