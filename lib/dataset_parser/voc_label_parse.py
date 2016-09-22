import xml.dom.minidom

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
        dom = xml.dom.minidom.parse(self.__file)
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        label = []
        for obj in objects:
            # Get name
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            # Get difficult
            difficult = obj.getElementsByTagName('difficult')[0].childNodes[0].data
            # Get bbox
            if len(obj.getElementsByTagName("bndbox")) == 0:
                print self.__file + "contains no object."
            for bboxs in obj.getElementsByTagName("bndbox"):
                xmin = bboxs.getElementsByTagName('xmin')[0].childNodes[0].data
                xmax = bboxs.getElementsByTagName('xmax')[0].childNodes[0].data
                ymin = bboxs.getElementsByTagName('ymin')[0].childNodes[0].data
                ymax = bboxs.getElementsByTagName('ymax')[0].childNodes[0].data
                label.append([classes_dict[name], int(difficult), int(float(xmin)),
                              int(float(xmax)), int(float(ymin)), int(float(ymax))])
        return label