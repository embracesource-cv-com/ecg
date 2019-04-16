# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/26 
@file: model_utils.py
@description:
"""
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from common import conf
import numpy as np
import keras.backend as K


def save_result(model, name, y_train, y_train_pred, y_vali, y_vali_pred):
    '''
    :param model: model object
    :param name: str, txt file name
    :param y_train,y_train_pred,y_vali,y_vali_pred: 1-D array
    :return:
    '''
    conf1 = pd.crosstab(y_train, y_train_pred)
    conf2 = pd.crosstab(y_vali, y_vali_pred)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    val_acc = metrics.accuracy_score(y_vali, y_vali_pred)
    train_report = classification_report(y_train, y_train_pred)
    val_report = classification_report(y_vali, y_vali_pred)
    with open(name, "w") as text_file:
        text_file.write('模型参数:' + str(model) + '\n')
        print('训练集准确率:' + "{0:.4%}".format(train_acc), file=text_file)
        print('验证集准确率:' + "{0:.4%}".format(val_acc), file=text_file)
        text_file.write('训练集混淆矩阵:' + '\n')
        text_file.write(str(conf1) + '\n')
        text_file.write('验证集混淆矩阵:' + '\n')
        text_file.write(str(conf2) + '\n')
        text_file.write('训练集report:' + '\n')
        text_file.write(str(train_report) + '\n')
        text_file.write('验证集report:' + '\n')
        text_file.write(str(val_report) + '\n')
    print(train_acc, val_acc, '\n', conf1, '\n', conf2, '\n', train_report, '\n', val_report)
    # return train_acc, val_acc, conf1, conf2, train_report, val_report


def evaluate(model, x_train, y_train, x_test, y_test, result_file):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    save_result(model, result_file, y_train, y_train_pred, y_test, y_test_pred)


def cal_f1_metric(model, x, y_true):
    y_prob = model.predict(x, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    if conf.one_hot:
        y_true = np.argmax(y_true, axis=1)
    f1 = f1_score(y_true, y_pred)
    f1 = np.round(f1, 3)
    return f1


def keras_f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


'''
def keras_f1_score(y_true, y_pred):
    """
    f1 score

    :param y_true:
    :param y_pred:
    :return:
    """
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))

'''
