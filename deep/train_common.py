# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/6 
@file: train_common.py
@description:
"""
from common import utils
from common.conf import current_config as conf
from common import pre_process
from deep.callback import log, lr_decay, ckpt_saver, Eval
from sklearn.model_selection import KFold
from common import model_utils
import numpy as np
from sklearn.utils import shuffle
from keras import Model
from keras.optimizers import Adam


def load_data():
    x = utils.load_dataset()
    x = x.transpose(0, 2, 1)
    y = utils.load_label()
    x, y = shuffle(x, y, random_state=conf.seed)
    x = pre_process.rm_noise_2d(signals=x, wave_name='bior2.6', level=8)
    print('x.shape,y.shape', x.shape, y.shape)
    return x, y


def compile_model(input_x, out):
    model = Model(inputs=input_x, outputs=out)
    model.summary()
    model.compile(optimizer=Adam(lr=conf.lr, decay=conf.weight_decay),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    if conf.continue_training:
        print('loading trained weights from..', conf.weights_to_transfer)
        model.load_weights(conf.weights_to_transfer, by_name=True)
    return model


def fit_model(x_train, y_train, x_test, y_test, model, model_index):
    print('number of training samples:', len(x_train))
    print('number of validation samples:', len(x_test))
    gen_train = utils.gen_no_random(x_train, y_train, conf.batch_size)
    val_accuracy = Eval(data=[x_test, y_test], mode='val data', interval=1)
    model.fit_generator(generator=gen_train,
                        validation_data=[x_test, y_test],
                        validation_steps=2,
                        steps_per_epoch=conf.steps_per_epoch,
                        epochs=conf.epochs,
                        callbacks=[log, lr_decay, ckpt_saver(model_index), val_accuracy])
    scores = model_utils.cal_f1_metric(model, x_test, y_test)
    return scores


def train_model(x, y, model_compiled):
    scores = []
    if not conf.ensemble:
        print(x.shape, y.shape)
        x_train, y_train, x_test, y_test = utils.split_data(x, y, train_ratio=conf.train_ratio)
        '''
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, '\n', y_train, '\n', y_test)
        if conf.lead_as_sample:
            x_train = x_train.reshape(-1, 5000, 1)
            x_test = x_test.reshape(-1, 5000, 1)
            y_train = np.repeat(y_train, 12)
            y_test = np.repeat(y_test, 12)
        '''
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, '\n', y_train, '\n', y_test)
        model = model_compiled
        score = fit_model(x_train, y_train, x_test, y_test, model, 0)
        print('F1 score of the model: ', score)
    else:
        kf = KFold(n_splits=conf.num_model)
        # sss = StratifiedShuffleSplit(n_splits=conf.num_model,test_size=1-conf.train_ratio,
        #                            random_state=conf.seed,)
        i = 1
        for train_index, test_index in kf.split(x, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_test)
            model = model_compiled
            score = fit_model(x_train, y_train, x_test, y_test, model, i)
            scores.append(score)
            i = i + 1
        scores = np.array(scores)
        print('F1 score of all model: ', scores)
        print('F1 score mean+/-std: ', "%0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))
