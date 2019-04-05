# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/12 
@file: callback.py
@description:
"""
import common.conf as conf
import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
import warnings
import os

log = TensorBoard(log_dir=conf.output_dir)
lr_decay = ReduceLROnPlateau(monitor='val_acc', patience=conf.patience, factor=0.95)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=50,verbose=0, mode='auto')

# ckpt_saver = ModelCheckpoint(filepath=conf.output_dir + 'weights.hdf5', verbose=1, save_best_only=True)

def ckpt_saver(model_index):
    return ModelCheckpoint(filepath=os.path.join(conf.output_dir, str(model_index) +'_weights.hdf5'),
                           verbose=1, monitor='val_acc',
                           save_best_only=True)


def evaluate(model, x, y_true):
    y_prob = model.predict(x, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    if conf.one_hot:
        y_true = np.argmax(y_true, axis=1)
    #acc = accuracy_score(y_true, y_pred)
    #conf_matrix = confusion_matrix(y_true, y_pred)
    #acc_report = classification_report(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('f1 score:',f1)
    #return acc, conf_matrix, acc_report, y_prob, y_pred


class Eval(keras.callbacks.Callback):
    def __init__(self, data, mode, interval=1):
        self.interval = interval
        self.x, self.y_true = data
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            evaluate(self.model, self.x, self.y_true)
            #acc, conf_matrix, acc_report,_,_ = evaluate(self.model, self.x, self.y_true)
            #print('For epoch' + str(epoch + 1) + ', the accuracy of ' + self.mode + ' is:\n', acc)
            #print('For epoch' + str(epoch + 1) + ', the confusion matrix of ' + self.mode + ' is:\n', conf_matrix)
            #print('For epoch' + str(epoch + 1) + ', the accuracy report of ' + self.mode + ' is:\n', acc_report)
