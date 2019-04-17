# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/8 
@file: verify_1d_conv.py
@description:
"""

from common.conf import current_config as conf
from deep.callback import evaluate
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model
import numpy as np
from train_1d_conv import lead_to_sample
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index


def vote_label(y_preds, i):
    unique_pred, counts = np.unique(y_preds[:, i], return_counts=True)
    label = unique_pred[np.argmax(counts)]
    return label


def ensemble_label(probs, preds, mode):
    ens_label = None
    if mode == 'label_vote':
        ens_label = [vote_label(preds, i) for i in range(preds.shape[1])]
    elif mode == 'prob_sum':
        sum_probs = np.sum(probs, axis=0)
        ens_label = np.argmax(sum_probs, axis=1)
    elif mode == 'prob_max':
        max_probs = np.max(probs, axis=0)
        ens_label = np.argmax(max_probs, axis=1)
    return ens_label


def ensemble_model():
    start_time = time.time()
    _, _, x_test, y_test = lead_to_sample()
    print('x_test.shape,y_test.shape', x_test.shape, y_test.shape)
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
    _, _, x_test, y_test = lead_to_sample()
    print('x_test.shape,y_test.shape', x_test.shape, y_test.shape)
    model_loaded = load_model(os.path.join(conf.output_dir, '0_weights.hdf5'))
    evaluate(model=model_loaded, x=x_test, y_true=y_test)


def vote_label_1s(y_preds, i):
    unique_pred, counts = np.unique(y_preds[i, :], return_counts=True)
    label = unique_pred[np.argmax(counts)]
    return label


def verify_single_model_1s():
    _, _, x_test, y_test = lead_to_sample()
    print('x_test.shape,y_test.shape', x_test.shape, y_test.shape)
    model_loaded = load_model(os.path.join(conf.output_dir, '0_weights.hdf5'))
    x_test = x_test.reshape([-1, 5000, 1])
    y_prob = model_loaded.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(y_pred.shape)
    y_pred = y_pred.reshape([-1, 12])
    y_pred = [vote_label_1s(y_pred, i) for i in range(y_pred.shape[0])]
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    if conf.ensemble:
        ensemble_model()
    elif conf.lead_as_sample:
        verify_single_model_1s()
