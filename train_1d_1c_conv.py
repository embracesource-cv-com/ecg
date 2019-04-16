# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/3
@file: train_1d_1c_conv.py
@description:
"""
from common import conf
from deep.base_nets import conv_1d_1c
from keras import Input
from deep.train_common import load_data, compile_model, train_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def choose_model():
    input_x = Input([conf.seq_len, conf.num_lead])
    out = conv_1d_1c.multiple_net(input_x)
    return input_x, out


def train():
    x, y = load_data()
    input_x, out = choose_model()
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


if __name__ == '__main__':
    train()
