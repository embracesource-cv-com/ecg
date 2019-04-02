# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/2 
@file: train_tradition.py
@description: python -W ignore train_tradition.py
"""
from extract_feature import get_all_feature
import utils
import tradition_model
import conf

signals = utils.load_data()
labels = utils.load_label()
all_feature = get_all_feature(signals)

x_train, y_train, x_test, y_test = utils.split_data(all_feature, labels, conf.train_ratio)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

if __name__ == '__main__':
    save_model = False
    print('fitting knn model..')
    result_file = 'knn01.txt'
    tradition_model.do_knn(x_train, y_train, x_test, y_test, save_model, result_file)

    print('fitting svm model..')
    result_file = 'svm01.txt'
    tradition_model.do_svm(x_train, y_train, x_test, y_test, save_model, result_file)

    print('fitting xgb model..')
    grid_search = False
    result_file = 'xgb01.txt'
    tradition_model.do_xgb(x_train, y_train, x_test, y_test, save_model, result_file, grid_search)
