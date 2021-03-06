# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/8 
@file: train_1d_1s_conv.py
@description:
"""

from common import conf
from deep.base_nets import conv_1d
from keras import Input
from deep.train_common import load_data, compile_model, train_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def choose_model():
    input_x = Input([conf.seq_len, conf.num_lead])
    model_conf = conf.model_1d
    if model_conf == 'simple_net':
        out = conv_1d.simple_net(input_x)
    if model_conf == 'ecg_resnet':
        out = conv_1d.ecg_resnet(input_x)
    if model_conf == 'mini_resnet':
        out = conv_1d.mini_resnet(input_x)
    return input_x, out


def train():
    x, y = load_data()
    input_x, out = choose_model()
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


if __name__ == '__main__':
    train()
