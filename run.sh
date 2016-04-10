#!/bin/bash

source ~/torch/install/bin/torch-activate
source /norep/Downloads/cuda/activate.sh
source /mldata-vpc/cudnn-v4/activate
CUDA_VISIBLE_DEVICES=0 th train.lua -model vgg_bn_drop -s logs/vgg --backend cudnn

