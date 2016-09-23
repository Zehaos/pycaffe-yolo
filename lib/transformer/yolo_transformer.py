import numpy as np
import random
import cv2

class YoloTransformer:

    """
    YoloTransformer is a class for preprocessing and deprocessing
    images for yolo.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0
        self.__flip = False
        self.__jitter_value = 0
        self.__dithering = False

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)

    def set_flip(self, flag):
        self.__flip = flag

    def flip(self, im, label):
        """
        Filp the im for augmentation.
        """
        assert self.__flip is True
        if random.choice((True, False)):
            im_fliped = np.fliplr(im)
            label_fliped = label
            obj_idx = np.where(label[:, :, 0] > 0)
            for [i, j] in zip(obj_idx[0], obj_idx[1]):
                label_fliped[i][j][1] = 1 - label[i][j][1]
            return (im_fliped, label_fliped)
        else:
            return (im, label)

    def set_jitter(self, jitter_value):
        assert jitter_value < 0.5
        self.__jitter_value = jitter_value

    def jitter(self, im, label):
        """
        Jitter the im for augmentation.
        """
        sh = random.uniform((-1)*self.__jitter_value, self.__jitter_value)  # Horizontal
        dh = sh * np.shape(im)[1]
        sv = random.uniform((-1) * self.__jitter_value, self.__jitter_value)  # Vertical
        dv = sv * np.shape(im)[0]
        translation_matrix = np.float32([[1, 0, dh], [0, 1, dv]])
        img_translated = cv2.warpAffine(im, translation_matrix, np.shape(im)[0:2])
        label_translated = np.zeros_like(label)
        label_translated[:] = label
        obj_idx = np.where(label[:, :, 0] > 0)
        for [i, j] in zip(obj_idx[0], obj_idx[1]):
            label_translated[i][j][1] = label[i][j][1] + sh
            label_translated[i][j][2] = label[i][j][2] + sv
        label_translated = self.__correct_label(im, label_translated)
        return (img_translated, label_translated)

    def __correct_label(self, im, label):
        """TODO: deal with small obj, and mass obj"""
        img_width = np.shape(im)[0]
        img_height = np.shape(im)[1]
        label_correct = np.zeros_like(label)
        obj_idx = np.where(label[:, :, 0] > 0)
        for [i, j] in zip(obj_idx[0], obj_idx[1]):
            label_ = label[i][j][:]
            x = label_[1]
            y = label_[2]
            w = label_[3]
            h = label_[4]
            xmax = img_width if (x * img_width + w * img_width * 0.5) > img_width \
                else (x * img_width + w * img_width * 0.5)
            ymax = img_height if (y * img_height + h * img_height * 0.5) > img_height \
                else (y * img_height + h * img_height * 0.5)
            xmin = 0 if (x * img_width - w * img_width * 0.5) < 0 \
                else (x * img_width - w * img_width * 0.5)
            ymin = 0 if (y * img_height - h * img_height * 0.5) < 0 \
                else (y * img_height - h * img_height * 0.5)
            if xmin > img_width or xmax < 0 or ymin > img_height or ymax < 0:
                continue
            else:
                label_correct[i][j][:] = label[i][j][:]
                label_correct[i][j][1] = 0.5 * (xmin + xmax) / img_width
                label_correct[i][j][2] = 0.5 * (ymin + ymax) / img_height
                label_correct[i][j][3] = (xmax - xmin) / img_width
                label_correct[i][j][4] = (ymax - ymin) / img_height
        return label_correct

    def set_color_dithering(self, flag):
        self.__dithering = flag

    def color_dithering(self, im):
        """
        Color dithering for data augmentation.
        Including brightness, contrast and saturation dithering.
        """
        contrast = random.gauss(1, 0.07)
        brightness = random.gauss(0, 5)
        saturation = random.gauss(0, 5)
        saturation_base = random.choice([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
        img = np.uint8(im*contrast + brightness + saturation_base*saturation)
        return img

