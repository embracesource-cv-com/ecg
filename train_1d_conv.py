# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/14 
@file: train_ecg.py
@description:
"""
import common.utils as utils
import common.conf as conf
from deep.base_nets import conv_1d, simple_model
from keras import Input
from keras import Model
from keras.optimizers import Adam
from deep.callback import log, lr_decay, ckpt_saver, Eval
from sklearn.model_selection import StratifiedShuffleSplit
from tradition.extract_feature import get_all_feature
import os
import numpy as np

'''
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf  # set GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)
'''
os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def load_data():
    x = utils.load_data()
    x = x.transpose(0, 2, 1)
    y = utils.load_label()
    return x, y


def complie_model():
    input_x = Input([conf.seq_len, conf.num_lead])
    out = simple_model.simple_net(input_x)
    # out = conv_1d.ecg_resnet(input_x)
    model = Model(inputs=input_x, outputs=out)
    model.summary()
    model.compile(optimizer=Adam(lr=conf.lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if conf.continue_training:
        print('loading trained weights from..', conf.weights_to_transfer)
        model.load_weights(conf.weights_to_transfer, by_name=True)
    return model


def fit_model(x_train, y_train, x_test, y_test, model, model_index):
    gen_train = utils.gen_no_random(x_train, y_train, conf.batch_size)
    gen_test = utils.generator(x_test, y_test, conf.batch_size)
    train_accuracy = Eval(data=next(gen_train), mode='training data', interval=1)
    val_sample_accuracy = Eval(data=next(gen_test), mode='val_sample data', interval=1)
    # x_test, y_test = x_test.reshape(-1, conf.seq_len,  conf.num_lead), y_test
    val_accuracy = Eval(data=[x_test, y_test], mode='val data', interval=1)
    model.fit_generator(generator=gen_train,
                        validation_data=[x_test, y_test],
                        validation_steps=2,
                        steps_per_epoch=conf.steps_per_epoch,
                        epochs=conf.epochs,
                        callbacks=[log, lr_decay, ckpt_saver(model_index)])


def train():
    x, y = load_data()
    if not conf.ensemble:
        x_train, y_train, x_test, y_test = utils.split_data(x, y, train_ratio=conf.train_ratio)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        model = complie_model()
        fit_model(x_train, y_train, x_test, y_test, model, 0)
    else:
        sss = StratifiedShuffleSplit(n_splits=conf.num_model, test_size=1 - conf.train_ratio, random_state=conf.seed)
        i = 1
        for train_index, test_index in sss.split(x, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('len(x_train), len(x_test):', len(x_train), len(x_test))
            model = complie_model()
            fit_model(x_train, y_train, x_test, y_test, model, i)
            i = i + 1


def train_extra_feature():
    print('获取1维心拍样本及标签...')
    file_names = utils.get_file_name()
    all_beats, all_labels = utils.load_data(file_names)
    x_train, y_train, x_test, y_test = utils.split_data(all_beats, all_labels, train_ratio=0.7)
    print('获取1维心拍的传统特征...')
    extra_x = get_all_feature()
    x_extra_train, _, x_extra_test, _ = utils.split_data(extra_x, all_labels, train_ratio=0.7)
    gen_train = utils.generator_extra([x_train, x_extra_train], y_train, conf.batch_size)
    gen_test = utils.generator_extra([x_test, x_extra_test], y_test, conf.batch_size)
    batch_x, batch_y = next(gen_train)
    print(len(batch_x), batch_x[0].shape, batch_x[1].shape, batch_y[0].shape)

    input_x = Input([conf.seq_len, 1])
    extra_x = Input([x_extra_train.shape[1]])
    main_out, auxiliary_out = conv_1d.resnet_with_extra_x(input_x, extra_x)
    model = Model(inputs=[input_x, extra_x], outputs=[main_out, auxiliary_out])
    model.summary()
    model.compile(optimizer=Adam(lr=conf.lr),
                  loss='sparse_categorical_crossentropy',
                  loss_weights=[1., 0.8],
                  metrics=['sparse_categorical_accuracy'])

    if conf.continue_training:
        print('loading trained weights from..', conf.weights_to_transfer)
        model.load_weights(conf.weights_to_transfer, by_name=True)

    model.fit_generator(generator=gen_train,
                        validation_data=gen_test,
                        validation_steps=2,
                        steps_per_epoch=conf.steps_per_epoch,
                        epochs=conf.epochs,
                        callbacks=[log, lr_decay, ckpt_saver(0)])


if __name__ == '__main__':
    if not conf.use_tradition_feature:
        train()
    else:
        train_extra_feature()
