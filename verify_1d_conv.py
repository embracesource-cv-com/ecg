# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/16 
@file: verify.py
@description:
"""

import common.conf as conf
import common.utils as utils
from deep.callback import evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import load_model
import numpy as np
import os
import sys
import time
from tradition.extract_feature import get_all_feature

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def load_data():
    print('获取样本及标签...')
    x = utils.load_data()
    x = x.transpose(0, 2, 1)
    y = utils.load_label()
    _, _, x_test, y_test = utils.split_data(x, y, train_ratio=conf.train_ratio)
    return x_test, y_test, x, y


def vote_label(y_preds, i):
    unique_pred, counts = np.unique(y_preds[:, i], return_counts=True)
    label = unique_pred[np.argmax(counts)]
    return label


def ensemble_label(probs, preds, mode):
    if mode == 'label_vote':
        ens_label = [vote_label(preds, i) for i in range(preds.shape[1])]
    if mode == 'prob_sum':
        sum_probs = np.sum(probs, axis=0)
        ens_label = np.argmax(sum_probs, axis=1)
    if mode == 'prob_max':
        max_probs = np.max(probs, axis=0)
        ens_label = np.argmax(max_probs, axis=1)
    return ens_label


def ensemble_model():
    start_time = time.time()
    _, _, x_test, y_test = load_data()
    model_names = [os.path.join(conf.output_dir, str(i + 1) + '_weights.hdf5') for i in range(conf.num_model)]
    # model_names = model_names[:3]
    accs = []
    y_probs = []
    y_preds = []
    for model in model_names:
        model_loaded = load_model(model)
        acc, conf_matrix, acc_report, y_prob, y_pred = evaluate(model=model_loaded, x=x_test, y_true=y_test)
        print('[%s] Accuracy: %1.5f' % (model, acc))
        print(acc_report)
        accs.append(acc)
        y_probs.append(y_prob)
        y_preds.append(y_pred)
    ens_label = ensemble_label(probs=np.array(y_probs), preds=np.array(y_preds), mode=conf.ensemble_mode)
    ens_acc = accuracy_score(y_test, ens_label)
    report = classification_report(y_test, ens_label)
    std, mean = np.std(accs), np.mean(accs)
    print('[%s] Accuracy: %1.5f+-%1.5f' % (str(conf.num_model) + ' single model', mean, std))
    print('[%s] Accuracy: %1.5f' % ('ensemble ' + str(conf.num_model) + ' models', ens_acc))
    print('evaluation report for all data', report)
    print("--- %s seconds ---" % (time.time() - start_time))


def verify_single_model():
    x_test, y_test, _, _ = load_data()
    model_loaded = load_model(os.path.join(conf.output_dir, '0_weights.hdf5'))
    acc, conf_matrix, acc_report, _, _ = evaluate(model=model_loaded, x=x_test, y_true=y_test)
    print('the accuracy of ' + ' is:\n', acc)
    print('the confusion matrix of ' + ' is:\n', conf_matrix)
    print('the accuracy report of ' + ' is:\n', acc_report)


def verify_extra_x():
    print('获取1维心拍的传统特征...')
    extra_x = get_all_feature()
    x_test, y_test, _, all_labels = load_data()
    x_extra_train, _, x_extra_test, _ = utils.split_data(extra_x, all_labels, train_ratio=0.7)
    val_x = [x_test.reshape(-1, conf.seq_len, conf.num_lead), x_extra_test]
    val_y = [y_test, y_test]

    model_loaded = load_model(os.path.join(conf.output_dir, 'weights.hdf5'))
    loss, loss_main, loss_aux, acc_main, acc_aux = model_loaded.evaluate(val_x, val_y)
    print(loss, loss_main, loss_aux, acc_main, acc_aux)
    predicted = model_loaded.predict(val_x, batch_size=64)
    predicted = predicted[0]
    predicted_label = np.argmax(predicted, axis=1)
    val_y = val_y[0]
    confusion_m = confusion_matrix(val_y, predicted_label)
    print('loss:', loss, '\n', 'accuracy:', acc_main, '\n', 'confusion matrix:\n', confusion_m)


if __name__ == '__main__':
    if not conf.use_tradition_feature:
        if conf.ensemble:
            print('yes')
            ensemble_model()
        else:
            verify_single_model()
    else:
        verify_extra_x()
