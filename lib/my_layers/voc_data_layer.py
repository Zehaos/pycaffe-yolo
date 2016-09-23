import scipy.misc
import caffe

import numpy as np
import os.path as osp

from random import shuffle

from transformer.simple_transformer import SimpleTransformer
from dataset_parser.voc_label_parse import VocLabelParser


class VOCLocDataLayerSyncSync(caffe.Layer):
    """
    This is a simple syncronous datalayer for training a yolo model on
    PASCAL.

    example:
    layer {
        name: "Data1"
        type: "Python"
        top: "Data1"
        top: "label"
        python_param {
            module: "voc_data_layer"
            layer: "VOCLocDataLayerSyncSync"
            param_str: "{
            \'list_root\': \'/path/to/list/root\',
             \'split\': \'train\',
             \'mean\': (104.00699, 116.66877, 122.67892),
             \'batch_size\': 16,
             \'im_shape\': (448,448),
             \'classes\': 20,
             \'coords\': 4,
             \'num\': 2,
             \'side\': 7
             }"
        }
    }
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # Yolo related params.
        self.classes = params.get('classes', 20)
        self.coords = params.get('coords', 4)
        self.side = params.get('side', 7)
        self.num = params.get('num', 7)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Yolo label shape
        top[1].reshape(self.batch_size, self.side, self.side, (self.coords+1) + self.classes)

        print_info("VOCLocDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, yolo_label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = yolo_label

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.mean = np.array(params['mean'])
        self.list_root = params['list_root']
        self.im_shape = params['im_shape']
        self.classes = params.get('classes', 20)
        self.coords = params.get('coords', 4)
        self.side = params.get('side', 7)
        self.num = params.get('num', 7)
        # get list of image indexes.
        list_file = params['split'] + '.txt'
        self.indexlist = [line.rstrip('\n') for line in open(
            osp.join(self.list_root, list_file))]
        shuffle(self.indexlist)
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(self.mean)

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file = index.split(' ')[0]
        im = np.asarray(scipy.misc.imread(image_file))
        self.ori_im_shape = np.shape(im)[0:2]
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        '''
        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1  # -1 or 1
        im = im[:, ::flip, :]'''

        # Load and prepare ground truth
        vocparser = VocLabelParser(index.split(' ')[1])
        anns = vocparser.parse()
        yolo_label = self.transform_to_yolo_labels(anns)
        # if flip == -1:  # do flip

        self._cur += 1
        return self.transformer.preprocess(im), yolo_label

    def transform_to_yolo_labels(self, labels):
        """
        Transform voc_label_parser' result to yolo label.
        :param labels: [is_obj, x, y, w, h, class_probs..], ...
        :return: yolo label
        """
        yolo_label = np.zeros([self.side, self.side, (1 + self.coords) + self.classes]).astype(np.float32)
        shuffle(labels)
        for label in labels:
            yolo_box = self.convert_to_yolo_box(self.ori_im_shape[::-1], list(label[2:]))
            assert np.max(yolo_box) < 1
            [loc_y, loc_x] = [int(np.floor(yolo_box[1] * self.side)), int(np.floor(yolo_box[0] * self.side))]
            yolo_label[loc_y][loc_x][0] = 1.0  # is obj
            yolo_label[loc_y][loc_x][1:5] = yolo_box  # bbox
            yolo_label[loc_y][loc_x][5:] = 0  # only one obj in one grid
            yolo_label[loc_y][loc_x][4+label[0]] = 1.0  # class
        return yolo_label

    def convert_to_yolo_box(self, size, box):
        """
        Convert VOC bbox to yolo bbox
        :param size: image shape
        :param box: voc bbox [xmin, xmax, ymin, ymax]
        :return: yolo bbox
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'list_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])