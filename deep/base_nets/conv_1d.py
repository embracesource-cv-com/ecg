# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/14 
@file: conv_1d.py
@description:
"""

from keras import Model, Input
import keras.layers as KL
import common.conf as conf
from keras.regularizers import l2
size = conf.conv_kernel_size


def conv_unit(x, filters, block, unit, trainable=True):
    filter1, filter2 = filters
    conv_name_base = 'conv_block' + str(block) + '_unit' + str(unit)
    bn_name_base = 'bn_block' + str(block) + '_unit' + str(unit)
    shortcut = x
    # bn-relu-dropout-conv-bn-relu-dropout-conv-bn
    x = KL.BatchNormalization(name=bn_name_base + '_1')(x, training=trainable)
    x = KL.Activation(activation='relu')(x)
    x = KL.Dropout(rate=conf.dropout_rate)(x)
    x = KL.Conv1D(filters=filter1, kernel_size=size, strides=1, padding='same', name=conv_name_base + '_1',
                  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = KL.BatchNormalization(name=bn_name_base + '_2')(x, training=trainable)
    x = KL.Activation(activation='relu', name='Prelu_block' + str(block) + '_unit' + str(unit))(x)
    x = KL.Dropout(rate=conf.dropout_rate)(x)
    x = KL.Conv1D(filters=filter2, kernel_size=size, strides=2, padding='same', name=conv_name_base + '_2',
                  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    shortcut = KL.Conv1D(filters=filter2, kernel_size=size, strides=2,
                         padding="same", name='conv_block' + str(block) + '_shortcut',
                         kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(shortcut)
    x = KL.Add(name=str(block) + '_unit' + str(unit) + '_add_shortcut')([x, shortcut])
    return x


def identity_unit(x, filters, block, unit, trainable=True):
    filter1, filter2 = filters
    conv_name_base = 'conv_block' + str(block) + '_unit' + str(unit)
    bn_name_base = 'bn_block' + str(block) + '_unit' + str(unit)
    shortcut = x
    # bn-relu-dropout-conv-bn-relu-dropout-conv-bn
    x = KL.BatchNormalization(name=bn_name_base + '_1')(x, training=trainable)
    x = KL.Activation(activation='relu')(x)
    x = KL.Dropout(rate=conf.dropout_rate)(x)
    x = KL.Conv1D(filters=filter1, kernel_size=size, strides=1, padding='same', name=conv_name_base + '_1',
                  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = KL.BatchNormalization(name=bn_name_base + '_2')(x, training=trainable)
    x = KL.Activation(activation='relu', name='Prelu_block' + str(block) + '_unit' + str(unit))(x)
    x = KL.Dropout(rate=conf.dropout_rate)(x)
    x = KL.Conv1D(filters=filter2, kernel_size=size, strides=1, padding='same', name=conv_name_base + '_2',
                  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    x = KL.Add(name=str(block) + '_unit' + str(unit) + '_add_shortcut')([x, shortcut])
    return x


def ecg_resnet(input_x):
    # head
    x = KL.Conv1D(filters=64, kernel_size=size, padding='same')(input_x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation(activation='relu')(x)
    # block 1
    x = conv_unit(x, filters=[64, 64], block=1, unit=1, trainable=True)
    x = identity_unit(x, filters=[64, 64], block=1, unit=2, trainable=True)
    # block2
    x = conv_unit(x, filters=[64, 64], block=2, unit=1, trainable=True)
    x = identity_unit(x, filters=[64, 64], block=2, unit=2, trainable=True)
    # block 3
    x = conv_unit(x, filters=[128, 128], block=3, unit=1, trainable=True)
    x = identity_unit(x, filters=[128, 128], block=3, unit=2, trainable=True)
    # block 4
    x = conv_unit(x, filters=[128, 128], block=4, unit=1, trainable=True)
    x = identity_unit(x, filters=[128, 128], block=4, unit=2, trainable=True)
    # block 5
    x = conv_unit(x, filters=[256, 256], block=5, unit=1, trainable=True)
    x = identity_unit(x, filters=[256, 256], block=5, unit=2, trainable=True)
    # block 6
    x = conv_unit(x, filters=[256, 256], block=6, unit=1, trainable=True)
    x = identity_unit(x, filters=[256, 256], block=6, unit=2, trainable=True)
    # block 7
    x = conv_unit(x, filters=[512, 512], block=7, unit=1, trainable=True)
    x = identity_unit(x, filters=[512, 512], block=7, unit=2, trainable=True)
    # block 8
    x = conv_unit(x, filters=[512, 512], block=8, unit=1, trainable=True)
    x = identity_unit(x, filters=[512, 512], block=8, unit=2, trainable=True)
    # tail
    x = KL.BatchNormalization(name='out_bn')(x, training=True)
    x = KL.Activation(activation='relu', name='out_relu')(x)
    x = KL.Flatten(name='out_flatten')(x)
    if not conf.use_tradition_feature:
        x = KL.Dense(units=conf.num_class, name='dense_soft_out', activation='softmax')(x)
    return x


def mini_resnet(input_x):
    # head
    x = KL.Conv1D(filters=64, kernel_size=size, padding='same')(input_x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation(activation='relu')(x)
    # block 1
    x = conv_unit(x, filters=[64, 64], block=1, unit=1, trainable=True)
    x = conv_unit(x, filters=[64, 64], block=2, unit=2, trainable=True)
    # block 3
    x = conv_unit(x, filters=[128, 128], block=3, unit=1, trainable=True)
    x = conv_unit(x, filters=[128, 128], block=4, unit=2, trainable=True)

    # block 5
    x = conv_unit(x, filters=[256, 256], block=5, unit=1, trainable=True)
    x = conv_unit(x, filters=[256, 256], block=6, unit=2, trainable=True)

    # block 7
    #x = conv_unit(x, filters=[512, 512], block=7, unit=1, trainable=True)
    #x = conv_unit(x, filters=[512, 512], block=8, unit=2, trainable=True)

    # tail
    x = KL.BatchNormalization(name='out_bn')(x, training=True)
    x = KL.Activation(activation='relu', name='out_relu')(x)
    #x = KL.Flatten(name='out_flatten')(x)
    x = KL.GlobalAveragePooling1D()(x)
    if not conf.use_tradition_feature:
        x = KL.Dense(units=conf.num_class, name='dense_soft_out', activation='softmax')(x)
    return x


def resnet_with_extra_x(input_x, extra_x):
    x = ecg_resnet(input_x=input_x)
    auxiliary_out = KL.Dense(units=conf.num_class, name='aux_out', activation='softmax')(x)
    extra_x = KL.Dense(units=128, name='extra_x_dense', activation='relu')(extra_x)
    extra_x = KL.Dropout(rate=conf.dropout_rate)(extra_x)
    combined = KL.concatenate([x, extra_x], name='extra_x_concat')
    combined = KL.Dense(units=128, name='combined_dense', activation='relu')(combined)
    combined = KL.Dropout(rate=conf.dropout_rate)(combined)
    main_out = KL.Dense(units=conf.num_class, name='main_out', activation='softmax')(combined)
    return main_out, auxiliary_out


if __name__ == '__main__':
    '''
    input_x = Input([conf.seq_len, 1])
    x = ecg_resnet(input_x=input_x)
    model = Model(inputs=input_x, outputs=x)
    model.summary()
    
    input_x = Input([conf.seq_len, 1])
    extra_x = Input([20])
    x = resnet_with_extra_x(input_x, extra_x)
    model = Model(inputs=[input_x, extra_x], outputs=x)
    model.summary()
    '''
