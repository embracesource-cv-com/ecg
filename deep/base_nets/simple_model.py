# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/2 
@file: simple_model.py
@description:
"""
from keras.layers import *
from keras.models import *
import common.conf as conf

def simple_net(nn_inputs):
    x = Conv1D(32, 16, strides=2,activation='relu',padding='same')(nn_inputs)
    x = Conv1D(32, 16,strides=2, activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, 16, strides=2,activation='relu',padding='same')(x)
    x = Conv1D(64, 16, strides=2,activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 16, strides=2,activation='relu',padding='same')(x)
    x = Conv1D(128, 16, strides=2,activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(256, 16, strides=2,activation='relu',padding='same')(x)
    x = Conv1D(256, 16, strides=2,activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(conf.dropout_rate)(x)
    #x = GlobalAveragePooling1D()(x)
    #x = Dropout(conf.dropout_rate)(x)
    x = Dense(conf.num_class, name='softmax_out',activation='softmax')(x)
    return x


if __name__ == '__main__':
    input_x = Input([conf.seq_len, conf.num_lead])
    out = simple_net(input_x)
    model = Model(inputs=input_x,outputs=out)
    model.summary()