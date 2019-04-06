# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/3 
@file: conv_1d_1c.py
@description:
"""

from keras.layers import *
from keras.models import *
import common.conf as conf
import keras.backend as K


def sub_net(nn_inputs):
    x = Conv1D(16, 16, strides=2, activation='relu', padding='same')(nn_inputs)
    x = Conv1D(16, 16, strides=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, 16, strides=2, activation='relu', padding='same')(x)
    x = Conv1D(32, 16, strides=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    # x = Conv1D(128, 16, strides=2, activation='relu', padding='same')(x)
    # x = Conv1D(128, 16, strides=2, activation='relu', padding='same')(x)
    # x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    return x


def slice_col(tensor_2d, index):
    tensor = tensor_2d[..., index]
    tensor = K.reshape(tensor, shape=[-1, conf.seq_len, 1])
    return tensor


def multiple_net(input):
    x1 = Lambda(function=slice_col, arguments={'index': 0})(input)
    x1 = sub_net(x1)
    x2 = Lambda(function=slice_col, arguments={'index': 1})(input)
    x2 = sub_net(x2)
    x3 = Lambda(function=slice_col, arguments={'index': 2})(input)
    x3 = sub_net(x3)
    x4 = Lambda(function=slice_col, arguments={'index': 3})(input)
    x4 = sub_net(x4)
    x5 = Lambda(function=slice_col, arguments={'index': 4})(input)
    x5 = sub_net(x5)
    x6 = Lambda(function=slice_col, arguments={'index': 5})(input)
    x6 = sub_net(x6)
    x7 = Lambda(function=slice_col, arguments={'index': 6})(input)
    x7 = sub_net(x7)
    x8 = Lambda(function=slice_col, arguments={'index': 7})(input)
    x8 = sub_net(x8)
    x9 = Lambda(function=slice_col, arguments={'index': 8})(input)
    x9 = sub_net(x9)
    x10 = Lambda(function=slice_col, arguments={'index': 9})(input)
    x10 = sub_net(x10)
    x11 = Lambda(function=slice_col, arguments={'index': 10})(input)
    x11 = sub_net(x11)
    x12 = Lambda(function=slice_col, arguments={'index': 11})(input)
    x12 = sub_net(x12)
    x = concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12])
    x = Dense(128, name='dense1', activation='relu')(x)
    x = Dropout(conf.dropout_rate)(x)
    x = Dense(conf.num_class, name='out', activation='softmax')(x)
    return x


if __name__ == '__main__':
    input_x = Input([conf.seq_len, conf.num_lead])
    out = multiple_net(input_x)
    model = Model(inputs=input_x, outputs=out)
    model.summary()
