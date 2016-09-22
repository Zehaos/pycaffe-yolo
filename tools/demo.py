"""
Thanks @author: xingw
"""

import _init_paths
import caffe
import numpy as np
import sys, getopt
import cv2
from transformer.simple_transformer import SimpleTransformer
import scipy.misc


def interpret_output(output, img_width, img_height):
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    w_img = img_width
    h_img = img_height
    print w_img, h_img
    threshold = 0.2
    iou_threshold = 0.2
    num_class = 20
    num_box = 2
    grid_size = 7
    probs = np.zeros((7, 7, 2, 20))
    class_probs = np.reshape(output[0:980], (7, 7, 20))
    #	print class_probs
    scales = np.reshape(output[980:1078], (7, 7, 2))
    #	print scales
    boxes = np.reshape(output[1078:], (7, 7, 2, 4))
    print boxes[3][3]
    offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= w_img
    boxes[:, :, :, 1] *= h_img
    boxes[:, :, :, 2] *= w_img
    boxes[:, :, :, 3] *= h_img

    for i in range(2):
        for j in range(20):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
    filter_mat_probs = np.array(probs >= threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0: continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2],
             boxes_filtered[i][3], probs_filtered[i]])

    return result


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def show_results(img, results, img_width, img_height):
    img_cp = img.copy()
    disp_console = True
    imshow = True
    #	if self.filewrite_txt :
    #		ftxt = open(self.tofile_txt,'w')
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]) // 2
        h = int(results[i][4]) // 2
        if disp_console: print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(
            int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(results[i][5])
        xmin = x - w
        xmax = x + w
        ymin = y - h
        ymax = y + h
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_width:
            xmax = img_width
        if ymax > img_height:
            ymax = img_height
        if imshow:
            cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            print xmin, ymin, xmax, ymax
            cv2.rectangle(img_cp, (xmin, ymin - 20), (xmax, ymin), (125, 125, 125), -1)
            cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (xmin + 5, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if imshow:
        cv2.imshow('YOLO detection', img_cp)
        cv2.waitKey(1000)


def main(argv):
    model_filename = ''
    weight_filename = ''
    img_filename = ''
    try:
        opts, args = getopt.getopt(argv, "hm:w:i:")
        print opts
    except getopt.GetoptError:
        print 'yolo_main.py -m <model_file> -w <output_file> -i <img_file>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'yolo_main.py -m <model_file> -w <weight_file> -i <img_file>'
            sys.exit()
        elif opt == "-m":
            model_filename = arg
        elif opt == "-w":
            weight_filename = arg
        elif opt == "-i":
            img_filename = arg
    print 'model file is "', model_filename
    print 'weight file is "', weight_filename
    print 'image file is "', img_filename

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(model_filename, weight_filename, caffe.TEST)
    img = caffe.io.load_image(img_filename)  # load the image using caffe io
    img_ = scipy.misc.imresize(img, (448, 448))
    transformer = SimpleTransformer([104.00699, 116.66877, 122.67892])
    input = transformer.preprocess(img_)
    out = net.forward_all(data=input)
    print out.iteritems()
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = interpret_output(out['result'][0], img.shape[1], img.shape[0])  # fc27 instead of fc12 for yolo_small
    show_results(img_cv, results, img.shape[1], img.shape[0])
    cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv[1:])