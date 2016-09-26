"""
Tools for testing.
"""
import _init_paths
import caffe
import numpy as np
import os
import Image, ImageDraw
from transformer.simple_transformer import SimpleTransformer
from transformer.yolo_transformer import YoloTransformer

classes_dict = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6,
                "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18,
                "train": 19, "tvmonitor": 20}

class NetTester:
    """
    Test code.
    """
    def __init__(self, use_gpu=True, model=[]):
        '''
        Init net.
        :param model: Network definition.
        '''
        if model == []:
            raise("model should not be empty!")
        print("Init NetTester: Use gpu: {}").format(use_gpu)
        print("Network: {}").format(model)
        if use_gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.__net = caffe.Net(model, caffe.TRAIN)

    def print_layers(self):
        print("Network layers:")
        for name, layer in zip(self.__net._layer_names, self.__net.layers):
            print("{:<15}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

    def print_blobs(self):
        print("Blobs:")
        for name, blob in self.__net.blobs.iteritems():
            print("{:<15}:  {}".format(name, blob.data.shape))

    def forward(self):
        self.__net.forward()

    def backward(self):
        self.__net.backward()

    def get_blob(self, blob_name):
        return self.__net.blobs[blob_name]

    def draw_label(self, image, label):
        img_shape = np.shape(image)
        mask = label[:, :, 0]
        locations = np.where(mask > 0)
        img = Image.fromarray(image)
        drawobj = ImageDraw.Draw(img)
        #print mask
        for [i, j] in zip(locations[0], locations[1]):
            l = label[i][j][:]
            yolo_box = l[1:5]
            x = yolo_box[0]
            y = yolo_box[1]
            w = yolo_box[2]
            h = yolo_box[3]
            width = w*img_shape[1]
            height = h*img_shape[0]
            xmin = int(x*img_shape[1] - 0.5*width)
            ymin = int(y*img_shape[0] - 0.5*height)
            xmax = int(xmin+width)
            ymax = int(ymin+height)
            drawobj.rectangle([xmin, ymin, xmax, ymax], outline="blue")
            drawobj.point([0.5*(xmin+xmax), 0.5*(ymin+ymax)])
            for k in range(0, 7):
                drawobj.line([448/7.0*k, 0, 448/7.0*k, 448])
                drawobj.line([0, 448 / 7.0 * k, 448, 448 / 7.0 * k])
            #print label[i][j]
        img.show()


net_tester = NetTester(True, "../models/googlenet/gnet_train.prototxt")
net_tester.print_layers()
net_tester.print_blobs()
net_tester.forward()

imgs_blob = net_tester.get_blob("data")
labels_blob = net_tester.get_blob("label")
img = imgs_blob.data[0]
label = labels_blob.data[0]

yolo_transformer = YoloTransformer((104.00699, 116.66877, 122.67892))
img = yolo_transformer.deprocess(img)
yolo_transformer.set_flip(False)
(img_fliped, label_fliped) = yolo_transformer.flip(img, label)
yolo_transformer.set_jitter(0)
(img_translated, label_translated) = yolo_transformer.jitter(img_fliped, label_fliped)
net_tester.draw_label(img_translated, label_translated)
yolo_transformer.set_color_dithering(False)
img_dithered = yolo_transformer.color_dithering(img_fliped)
net_tester.draw_label(img_dithered, label_fliped)