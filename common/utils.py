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
from common.conf import current_config as conf
import scipy.io as sio
import math
import random
import numpy as np


def re_create_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_file_name(data_type):
    path = os.path.join(conf.data_path, data_type)
    file_names = glob(path + '/*.' + conf.data_suffix)
    file_names.sort()
    return file_names


def load_dataset(data_type='TRAIN'):
    file_names = get_file_name(data_type)
    signals = []
    for file in file_names:
        signal = sio.loadmat(file)['data']
        signals.append(signal)
    signals = np.array(signals)
    return signals


def load_label():
    labels = pd.read_csv(conf.label_path, header=None, sep='\t')
    labels = np.array(labels[1])
    return labels


def seg_signal(x, seq_len, seg_len):
    start = np.arange(0, seq_len, seg_len)
    end = np.append(start, seq_len)
    end = np.delete(end, 0)
    num_part = len(start)
    x_seg = np.empty(shape=[len(x), seg_len, num_part])
    for i in range(num_part):
        seg_part = x[:, start[i]:end[i]]
        x_seg[:, :, i] = seg_part
    return x_seg


def split_data(all_x, all_y, train_ratio):
    random.seed(conf.seed)
    num_samples = len(all_x)
    train_size = int(num_samples * train_ratio)
    train = random.sample(range(num_samples), train_size)  # 无重复
    train = np.array(train)
    test = np.array([i for i in range(num_samples) if i not in train])
    x_train = all_x[train]
    y_train = all_y[train]
    x_test = all_x[test]
    y_test = all_y[test]
    return x_train, y_train, x_test, y_test


def generator(x, y, batch_size):
    while True:
        index = np.random.randint(0, len(y), batch_size)
        yield x[index], y[index]


def gen_no_random(x, y, batch_size):
    steps = math.floor(len(x) / batch_size)
    print('steps:', steps)
    while True:
        for i in range(steps):
            index = range(i * batch_size, i * batch_size + batch_size)
            yield x[index], y[index]
