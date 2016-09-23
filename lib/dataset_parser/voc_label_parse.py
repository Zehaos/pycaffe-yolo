#import xml.dom.minidom
from xml.etree import ElementTree as ET

classes_dict = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6,
                "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18,
                "train": 19, "tvmonitor": 20}

class VocLabelParser:
    """
    PASCAL VOC label parser for voc_data_layer.py.

    example:
        parser = VocLabelParser('/path/to/xml/file')
        # label : [[class, difficult, xmin, xmax, ymin, ymax], ...]
        label = parser.parse()
    """
    __file = ''

    def __init__(self, anno_file):
        self.__file = anno_file

    def parse(self):
        """
        :return: [[class, difficult, xmin, xmax, ymin, ymax], ...]
        """
        tree = ET.parse(self.__file)
        root = tree.getroot()
        labels = []
        for object in root.findall('object'):
            name = object.find('name').text
            difficult = object.find('difficult').text
            xmin =  object.find('bndbox').find('xmin').text
            xmax = object.find('bndbox').find('xmax').text
            ymin = object.find('bndbox').find('ymin').text
            ymax = object.find('bndbox').find('ymax').text
            labels.append([classes_dict[name], int(difficult), int(float(xmin)),
                          int(float(xmax)), int(float(ymin)), int(float(ymax))])
        return labels

parser = VocLabelParser('/home/zehao/Dataset/VOC-DATASET/VOCdevkit/VOC2012/Annotations/2007_007763.xml')
labels = parser.parse()
print labels