# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/22 
@file: utils.py
@description:
"""
from glob import glob
import os
import pandas as pd
import shutil
import common.conf as conf
import scipy.io as sio
import numpy as np


def re_create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_file_name():
    path = os.path.join(conf.data_path, 'TRAIN')
    file_names = glob(path + '/*.' + conf.data_suffix)
    file_names.sort()
    return file_names


def load_data():
    file_names = get_file_name()
    signals = []
    for file in file_names:
        signal = sio.loadmat(file)['data']
        signals.append(signal)
    signals = np.array(signals)
    print('signals.shape:', signals.shape)
    return signals


def load_label():
    labels = pd.read_csv(conf.label_path, header=None, sep='\t')
    labels = np.array(labels[1])
    return labels


def split_data(all_x, all_y, train_ratio):
    np.random.seed(2019)
    train = np.random.choice([True, False], len(all_y), replace=True, p=[train_ratio, 1 - train_ratio])
    x_train = all_x[train]
    x_test = all_x[~train]
    y_train = all_y[train]
    y_test = all_y[~train]
    return x_train, y_train, x_test, y_test


def generator(x, y, batch_size):
    while True:
        np.random.seed(2019)
        index = np.random.randint(0, len(y), batch_size)
        yield x[index].reshape([batch_size, -1, 1]), y[index]