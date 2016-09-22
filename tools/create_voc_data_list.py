import os
import random
import math


class ListCreator:
    """
    List creator for data preparation that needs a pair of image & label.

    example:
        # [0.8, 0.1, 0.1] is the quota of train, test, validation component.
        creator = ListCreator([0.8, 0.1, 0.1])
        #Label folder is corresponding to image folder. You can add more as you like.
        creator.add_path('/paht/to/image/file', '/paht/to/label/file')
        creator.add_path('/paht/to/image/file', '/paht/to/label/file')
        creator.create()
    """
    __path = []
    __split = []
    __img_label_pair = []

    def __init__(self, split=[]):
        """
        list_creator constructor.
        :param split: proportion of train, val, test, eg.[0.5, 0.1, 0.4]
        """
        if len(split) > 3:
            raise Exception("Length of split should be less than 3.")
        self.__split = split

    def add_path(self, image_path, label_path):
        self.__path.append([image_path, label_path])

    def create(self):
        for path_pair in self.__path: # Create image list
            for image_file in os.listdir(path_pair[0]):
                if os.path.splitext(image_file)[1] in ['.jpg', '.png', '.bmp']:
                    if os.path.exists(path_pair[1] + '/' + os.path.splitext(image_file)[0]+'.xml'):
                        self.__img_label_pair.append([path_pair[0] + '/' + image_file,
                                                      path_pair[1] + '/' + os.path.splitext(image_file)[0]+'.xml'])
                    else:
                        print(image_file + " has no corresponding label file.")

        random.shuffle(self.__img_label_pair)

        if len(self.__split) == 0:  # Only create train list.
            self.__write('train', self.__img_label_pair)
        else:
            index_last = 0
            for s in zip(self.__split, ['train', 'test', 'validate']):
                index_first = index_last
                index_last += int(math.floor(s[0]*len(self.__img_label_pair)))
                try:
                    print index_last, index_first
                    self.__write(s[1], self.__img_label_pair[index_first:index_last])
                except Exception, e:
                    print Exception, ":", e

    def __write(self, split, lines):
        try:
            output = open(split + '.txt', 'w')
        except Exception, e:
            print Exception, ":", e
        for line in lines:
            output.write(line[0] + ' ' + line[1] + '\n')
        output.close()
        return True

VOC_2007_img = '/home/zehao/Dataset/VOC-DATASET/VOCdevkit/VOC2007/JPEGImages'
VOC_2012_img = '/home/zehao/Dataset/VOC-DATASET/VOCdevkit/VOC2012/JPEGImages'
VOC_2007_label = '/home/zehao/Dataset/VOC-DATASET/VOCdevkit/VOC2007/Annotations'
VOC_2012_label = '/home/zehao/Dataset/VOC-DATASET/VOCdevkit/VOC2012/Annotations'

list_creator = ListCreator()
list_creator.add_path(VOC_2007_img, VOC_2007_label)
list_creator.add_path(VOC_2012_img, VOC_2012_label)
list_creator.create()


