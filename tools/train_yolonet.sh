#!/usr/bin/env sh

export PYTHONPATH=../lib:../caffe-yolo/python

CAFFE_HOME=../caffe-yolo

SOLVER=../models/yolonet/yolonet_solver.prototxt
WEIGHTS=../pretrain_weights/extraction_convs20.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0

