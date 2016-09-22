"""
Script for create yolonet model.
"""
from __future__ import print_function
from caffe import layers as L, params as P, to_proto

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group,
                         param=dict(lr_mult=0),
                         weight_filler=dict(type='gaussian', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    return conv, L.ReLU(conv, in_place=True,
                        relu_param=dict(negative_slope=0.1))

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,
                        param=dict(lr_mult=0),
                        weight_filler=dict(type='gaussian', std=0.01),
                        bias_filler=dict(type='constant', value=0))
    return fc, L.ReLU(fc, in_place=True,
                      relu_param=dict(negative_slope=0.1))

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def yolonet():
    # Python data layer
    pydata_params = dict(list_root='/home/zehao/WorkSpace/caffe/examples/yolo/lists')
    pydata_params['split'] = 'train'
    pydata_params['mean'] = (104.00699, 116.66877, 122.67892)
    pydata_params['batch_size'] = 16
    pydata_params['im_shape'] = (448, 448)
    pydata_params['classes'] = 20
    pydata_params['coords'] = 4
    pydata_params['num'] = 2
    pydata_params['side'] = 7
    pylayer = 'VOCLocDataLayerSyncSync'
    data, label = L.Python(module='voc_data_layer',name='DataLayer', layer=pylayer,
                            ntop=2, param_str=str(pydata_params))

    # the net itself
    conv1, relu1 = conv_relu(data, 7, 64, stride=2, pad=3)
    pool1 = max_pool(relu1, 2, stride=2)

    conv2, relu2 = conv_relu(pool1, 3, 192, stride=1, pad=1)
    pool2 = max_pool(relu2, 2, stride=2)

    conv3, relu3 = conv_relu(pool2, 1, 128, stride=1, pad=0)
    conv4, relu4 = conv_relu(relu3, 3, 256, stride=1, pad=1)
    conv5, relu5 = conv_relu(relu4, 1, 256, stride=1, pad=0)
    conv6, relu6 = conv_relu(relu5, 3, 512, stride=1, pad=1)
    pool6 = max_pool(relu6, 2, stride=2)

    conv7, relu7 = conv_relu(pool6, 1, 256, stride=1, pad=0)
    conv8, relu8 = conv_relu(relu7, 3, 512, stride=1, pad=1)
    conv9, relu9 = conv_relu(relu8, 1, 256, stride=1, pad=0)
    conv10, relu10 = conv_relu(relu9, 3, 512, stride=1, pad=1)
    conv11, relu11 = conv_relu(relu10, 1, 256, stride=1, pad=0)
    conv12, relu12 = conv_relu(relu11, 3, 512, stride=1, pad=1)
    conv13, relu13 = conv_relu(relu12, 1, 256, stride=1, pad=0)
    conv14, relu14 = conv_relu(relu13, 3, 512, stride=1, pad=1)

    conv15, relu15 = conv_relu(relu14, 1, 512, stride=1, pad=0)
    conv16, relu16 = conv_relu(relu15, 3, 1024, stride=1, pad=1)
    pool16 = max_pool(relu16, 2, stride=2)

    conv17, relu17 = conv_relu(pool16, 1, 512, stride=1, pad=0)
    conv18, relu18 = conv_relu(relu17, 3, 1024, stride=1, pad=1)
    conv19, relu19 = conv_relu(relu18, 1, 512, stride=1, pad=0)
    conv20, relu20 = conv_relu(relu19, 3, 1024, stride=1, pad=1)

    conv21, relu21 = conv_relu(relu20, 3, 1024, stride=1, pad=1)
    conv22, relu22 = conv_relu(relu21, 3, 1024, stride=2, pad=1)

    conv23, relu23 = conv_relu(relu22, 3, 1024, stride=1, pad=1)
    conv24, relu24 = conv_relu(relu23, 3, 1024, stride=1, pad=1)

    fc25, relu25 = fc_relu(relu24, 4096)
    result = L.InnerProduct(relu25, num_output=1470,
                        weight_filler=dict(type='gaussian', std=0.01),
                        bias_filler=dict(type='constant', value=0))
    # Python loss layer
    pydata_params = dict(classes=20)
    pydata_params['coords'] = 4
    pydata_params['side'] = 7
    pydata_params['num'] = 2
    pydata_params['object_scale'] = 1
    pydata_params['noobject_scale'] = 0.5
    pydata_params['class_scale'] = 1
    pydata_params['coord_scale'] = 5
    pydata_params['sqrt'] = True
    pylayer = 'YoloLossLayer'
    loss = L.Python(result, label, name='YoloLoss', module='yolo_loss_layer', layer=pylayer,
                            ntop=1, param_str=str(pydata_params))
    return to_proto(loss)

def make_net():
    with open('./yolonet_train.prototxt', 'w') as f:
        print(yolonet(), file=f)

    with open('./yolonet_test.prototxt', 'w') as f:
        print(yolonet(), file=f)

if __name__ == '__main__':
    make_net()