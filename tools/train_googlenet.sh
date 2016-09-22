#!/usr/bin/env sh

export PYTHONPATH=../lib:../caffe-yolo/python

CAFFE_HOME=../caffe-yolo

SOLVER=../models/googlenet/gnet_solver.prototxt
WEIGHTS=../pretrain_weights/bvlc_googlenet.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0

