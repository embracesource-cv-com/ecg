# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/7 
@file: conv_lstm.py
@description:
"""

import keras.layers as KL
from keras import Model
from common.conf import current_config as conf


def conv_lstm(x):
    x = KL.Conv1D(filters=40, kernel_size=5, strides=1)(x)
    x = KL.MaxPool1D(pool_size=2, strides=2)(x)
    x = KL.Conv1D(filters=32, kernel_size=3, strides=1)(x)
    x = KL.MaxPool1D(pool_size=2, strides=2)(x)
    x = KL.LSTM(units=32, recurrent_dropout=0.25, dropout=0.5, return_sequences=True)(x)
    x = KL.LSTM(units=16, recurrent_dropout=0.25, return_sequences=True)(x)
    x = KL.LSTM(units=4, return_sequences=False)(x)
    x = KL.Dense(units=conf.num_class, activation='softmax')(x)
    return x


if __name__ == '__main__':
    x_in = KL.Input([211, 24])
    print(x_in)
    x_out = conv_lstm(x_in)
    model = Model(inputs=x_in, outputs=x_out)
    model.summary()
