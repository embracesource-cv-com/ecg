# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/7 
@file: train_2d_conv.py
@description:
"""

from common.conf import current_config as conf
from deep.base_nets import simple_net, resnet
from keras import Input
from deep.train_common import load_data, compile_model, train_model
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def choose_model():
    input_x = Input([conf.num_lead, conf.seq_len, 1])
    model_conf = conf.model_2d
    if model_conf == 'simple_net':
        out = simple_net.simple_net(input_x)
    elif model_conf == 'resnet':
        out = resnet.resnet(input_x, 'resnet50', False, True)
    return input_x, out


def train():
    x, y = load_data()
    x = x.transpose([0, 2, 1])
    x = np.expand_dims(x, 3)
    # x = np.tile(x,[1,1,3])
    print(x.shape)
    input_x, out = choose_model()
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


if __name__ == '__main__':
    train()
