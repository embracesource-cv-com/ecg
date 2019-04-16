# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/14 
@file: train_ecg.py
@description:
"""
from common import conf
from deep.base_nets import conv_1d
from keras import Input
from deep.train_common import load_data, compile_model, train_model
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def choose_model(input_x):
    model_conf = conf.model_1d
    if model_conf == 'simple_net':
        out = conv_1d.simple_net(input_x)
    if model_conf == 'ecg_resnet':
        out = conv_1d.ecg_resnet(input_x)
    if model_conf == 'mini_resnet':
        out = conv_1d.mini_resnet(input_x)
    return out


def train_tmp():
    x, y = load_data()
    input_x = Input([conf.seq_len, conf.num_lead])
    out = choose_model(input_x)
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


def train():
    x, y = load_data()
    x = x.reshape(-1, 5000, 1)
    y = np.repeat(y, 12)
    print('x.shape,y.shape', x.shape, y.shape)
    input_x = Input([conf.seq_len, 1])
    out = choose_model(input_x)
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


if __name__ == '__main__':
    train()
