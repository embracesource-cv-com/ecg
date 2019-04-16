# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/8 
@file: train_conv_lstm.py
@description:
"""

from common import conf
from deep.base_nets import conv_lstm
from keras import Input
from deep.train_common import load_data, compile_model, train_model
import os
import numpy as np
from common import utils


def choose_model(input_x):
    out = conv_lstm.conv_lstm(input_x)
    return out


def train():
    x, y = load_data()
    x = x[:, :, 1]
    print(x.shape)
    x = utils.seg_signal(x, conf.seq_len, conf.seg_len)
    print('x_seg.shape:', x.shape)
    channel = int(conf.seq_len / conf.seg_len)
    input_x = Input([conf.seg_len, channel])
    out = choose_model(input_x)
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


def train_tmp():
    x, y = load_data()
    input_x = Input([5000, 12])
    out = choose_model(input_x)
    model_compiled = compile_model(input_x, out)
    train_model(x, y, model_compiled)


if __name__ == '__main__':
    train()
