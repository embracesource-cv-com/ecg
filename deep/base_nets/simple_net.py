# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2018/12/25 
@file: base_net.py
@description: defined simple net, resnet, resnet-ir
"""
from keras.layers import *
import keras.layers as KL
from keras import Model
from common.conf import current_config as conf

size = conf.conv_kernel_size


def simple_net(nn_inputs):
    x = Conv2D(32, (3, size), activation='relu', padding='same')(nn_inputs)
    x = Conv2D(32, (3, size), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 4))(x)

    x = Conv2D(16, (3, size), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, size), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)

    x = Conv2D(32, (3, size), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, size), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    # x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(conf.dropout_rate)(x)
    x = Dense(conf.num_class, name='softmax_out', activation='softmax')(x)
    return x


if __name__ == '__main__':
    input_shape = (12, 5000, 1)
    nn_input = KL.Input(shape=input_shape)
    embedding = simple_net(nn_inputs=nn_input)
    model = Model(inputs=nn_input, outputs=embedding)
    print(model.summary())
