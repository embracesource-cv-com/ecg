# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/3 
@file: train_1d_1c_conv.py
@description:
"""
import common.utils as utils
import common.conf as conf
from deep.base_nets import conv_1d, conv_1d_simple, conv_1d_1c
from keras import Input
from keras import Model
from keras.optimizers import Adam
from deep.callback import log, lr_decay, ckpt_saver, Eval
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from common import model_utils
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def load_data():
    x = utils.load_data()
    x = x.transpose(0, 2, 1)
    y = utils.load_label()
    return x, y


def complie_model():
    input_x = Input([conf.seq_len, conf.num_lead])
    out = conv_1d_1c.multiple_net(input_x)
    model = Model(inputs=input_x, outputs=out)
    model.summary()
    model.compile(optimizer=Adam(lr=conf.lr,decay=conf.weight_decay),
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
    val_accuracy = Eval(data=[x_test, y_test], mode='val data', interval=1)
    model.fit_generator(generator=gen_train,
                        validation_data=[x_test, y_test],
                        validation_steps=2,
                        steps_per_epoch=conf.steps_per_epoch,
                        epochs=conf.epochs,
                        callbacks=[log, lr_decay, ckpt_saver(model_index), val_accuracy])
    scores = model_utils.cal_f1_metric(model, x_test, y_test)
    return scores


def train():
    scores = []
    x, y = load_data()
    print(x.shape,y.shape)
    if not conf.ensemble:
        print('conf.train_ratio:', conf.train_ratio)
        x_train,x_test, y_train,  y_test = train_test_split(x, y, test_size=1 - conf.train_ratio, random_state=0)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        model = complie_model()
        score = fit_model(x_train, y_train, x_test, y_test, model, 0)
        print('F1 score of the model: ', score)
    else:
        sss = StratifiedShuffleSplit(n_splits=conf.num_model, test_size=1 - conf.train_ratio, random_state=conf.seed)
        i = 1
        for train_index, test_index in sss.split(x, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('len(x_train), len(x_test):', len(x_train), len(x_test))
            model = complie_model()
            score = fit_model(x_train, y_train, x_test, y_test, model, i)
            scores.append(score)
            i = i + 1
        scores = np.array(scores)
        print('F1 score of all model: ', scores)
        print('F1 score mean+/-std: ', "%0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    if not conf.use_tradition_feature:
        train()
