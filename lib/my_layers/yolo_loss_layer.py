import caffe
import numpy as np


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def box_rmse(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))


def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    return w * h


def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u


def overlap(x1, w1, x2, w2):
    left = max(x1 - w1 / 2., x2 - w2 / 2.)
    right = min(x1 + w1 / 2., x2 + w2 / 2.)
    return right - left


class YoloLossLayer(caffe.Layer):
    """
    Compute the YOLO Loss.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute yolo loss.")
        # config
        params = eval(self.param_str)
        self.classes = params.get('classes', 20)
        self.coords = params.get('coords', 4)
        self.side = params.get('side', 7)
        self.num = params.get('num', 2)
        self.object_scale = params.get('object_scale', 1)
        self.noobject_scale = params.get('noobject_scale', 0.5)
        self.class_scale = params.get('class_scale', 1)
        self.coord_scale = params.get('coord_scale', 5)
        self.sqrt = params.get('sqrt', True)
        self.batch_size = np.shape(bottom[0].data)[0]

    def reshape(self, bottom, top):
        # check input dimensions
        if bottom[0].count != np.shape(bottom[0].data)[0]\
                *self.side*self.side*((self.coords + 1)*self.num + self.classes):
            raise Exception("Length of Bottom[0] not equal to batch*side*side*((coords+1)*num+classes).")
        if bottom[1].count != np.shape(bottom[1].data)[0]\
                *self.side*self.side*((self.coords + 1) + self.classes):
            raise Exception("Length of Bottom[0] not equal to batch*side*side*((coords+1)+classes).")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.avg_iou = 0
        self.avg_cat = 0
        self.avg_allcat = 0
        self.avg_obj = 0
        self.avg_anyobj = 0
        count = 0

        locations = np.where(np.zeros([self.side, self.side]) == 0)
        for b in range(self.batch_size):
            obj_locations = np.where(bottom[1].data[b, :, :, 0] > 0)
            for (i, j) in zip(locations[0], locations[1]):
                loc_idx = i * self.side + j
                for n in range(self.num):
                    obj_idx = self.side * self.side * self.classes + loc_idx * self.num + n
                    self.avg_anyobj += bottom[0].data[b][obj_idx]
                    self.diff[b][obj_idx] = self.noobject_scale * bottom[0].data[b][obj_idx]
                best_index = -1
                best_iou = 0.
                best_rmse = 20.
                if (i, j) in zip(obj_locations[0], obj_locations[1]):  # has obj
                    class_index = loc_idx * self.classes
                    truth_probs = bottom[1].data[b, i, j, 5:]
                    out_probs = bottom[0].data[b, class_index:(class_index+self.classes)]
                    self.diff[b][class_index:(class_index+self.classes)] = self.class_scale * (out_probs - truth_probs)
                    self.avg_cat += out_probs[np.where(truth_probs == 1)]
                    self.avg_allcat += np.sum(out_probs)

                    truth_box_ = np.zeros(4)
                    truth_box_[:] = bottom[1].data[b, i, j, 1:5]

                    truth_box_[0] = truth_box_[0]*self.side - j
                    truth_box_[1] = truth_box_[1]*self.side - i
                    truth_box_[0] /= self.side
                    truth_box_[1] /= self.side
                    truth_box = np.zeros(4)
                    truth_box[:] = bottom[1].data[b, i, j, 1:5]
                    out_boxs = []
                    for n in range(self.num):
                        box_index = self.side*self.side * (self.classes + self.num) + (loc_idx * self.num + n)*self.coords
                        out_box = np.zeros(4)
                        out_box[:] = bottom[0].data[b, box_index:(box_index+self.coords)]
                        out_box[0] /= self.side
                        out_box[1] /= self.side
                        if self.sqrt:
                            out_box[2] **= 2
                            out_box[3] **= 2
                        out_boxs.append(out_box)
                        iou = box_iou(out_box, truth_box_)
                        rmse = box_rmse(out_box, truth_box_)
                        if best_iou > 0 or iou > 0:
                            if iou > best_iou:
                                best_iou = iou
                                best_index = n
                        else:
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_index = n
                        assert best_index != -1
                    iou = box_iou(out_boxs[best_index], truth_box_)

                    box_index = self.side*self.side * (self.classes + self.num) + (loc_idx * self.num + best_index) * self.coords
                    obj_idx = self.side * self.side * self.classes + loc_idx * self.num + best_index
                    self.avg_obj += bottom[0].data[b][obj_idx]
                    #rescore
                    self.diff[b][obj_idx] = self.object_scale * (bottom[0].data[b][obj_idx] - iou)
                    if self.sqrt:
                        truth_box_[2] = np.sqrt(truth_box_[2])
                        truth_box_[3] = np.sqrt(truth_box_[3])
                    self.diff[b][box_index:box_index+self.coords] = \
                        self.coord_scale*(bottom[0].data[b][box_index:box_index+self.coords] - truth_box)
                    if self.sqrt:
                        self.diff[b][box_index+2:box_index + self.coords] = \
                            self.coord_scale * (bottom[0].data[b][box_index+2:box_index + self.coords] - truth_box_[2:4])
                    self.avg_iou += iou
                    count += 1
        print np.sum(np.power(self.diff, 2))
        top[0].data[...] = np.sum(np.power(self.diff, 2))
        print("Detection Avg IOU:{}, Pos Cat:{}, All Cat:{}, Pos Obj:{}, Any Obj:{}, count:{}")\
            .format(self.avg_iou/count, self.avg_cat[0]/count, self.avg_allcat/(count*self.classes),
                    self.avg_obj/count, self.avg_anyobj/(self.batch_size*self.side*self.side*self.num), count)

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
