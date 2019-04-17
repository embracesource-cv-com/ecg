# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/1 
@file: tradition_model.py
@description:
"""
from common.conf import current_config as conf
from sklearn import svm
import time
from common.model_utils import evaluate
import pickle
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import metrics
import os


def do_svm(x_train, y_train, x_test, y_test, save_model, result_file):
    # kernel = ['Linear','poly','rbf']
    kernel = ['Linear', 'rbf']
    # svm.SVC(kernel='poly'
    clf = [svm.LinearSVC(), svm.SVC(kernel='rbf')]

    train_acc = []
    vali_acc = []
    for i in range(len(clf)):
        start_time = time.time()
        clf[i].fit(x_train, y_train)
        y_train_pred = clf[i].predict(x_train)
        y_test_pred = clf[i].predict(x_test)
        train_acc.append(metrics.accuracy_score(y_train, y_train_pred))
        vali_acc.append(metrics.accuracy_score(y_test, y_test_pred))
        print('---' + kernel[i] + '_kernel---' + '\n' + "--- %s min ---" % ((time.time() - start_time) / 60))

    # save best model
    m = max(vali_acc)
    max_loc = [i for i, j in enumerate(vali_acc) if j == m]
    i = max_loc[0]
    model = clf[i]
    if save_model:
        pickle.dump(model, open(os.path.join(conf.output_dir, "svm.pickle.dat"), "wb"))

    # save model performace
    evaluate(model, x_train, y_train, x_test, y_test, result_file)


def do_knn(x_train, y_train, x_test, y_test, save_model, result_file):
    model = KNeighborsClassifier(n_neighbors=conf.num_class)
    model.fit(x_train, y_train)
    if save_model:
        pickle.dump(model, open(os.path.join(conf.output_dir, "knn.pickle.dat"), "wb"))

    # save model performace
    evaluate(model, x_train, y_train, x_test, y_test, result_file)


def do_xgb(x_train, y_train, x_test, y_test, save_model, result_file, grid_search):
    if grid_search:
        gbm_params = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 300, 500],
            'max_depth': [2, 3, 6]
        }
        gbm = xgboost.XGBClassifier()
        cv = StratifiedKFold(y_train)
        grid = GridSearchCV(gbm, gbm_params, cv=cv)
        grid.fit(x_train, y_train)
        model = grid.best_estimator_
        print(model)
    else:
        model = xgboost.XGBClassifier()
        model.fit(x_train, y_train)

    # save model to file
    if save_model:
        pickle.dump(model, open(os.path.join(conf.output_dir, "doXgb.pickle.dat"), "wb"))
    # save model performace
    evaluate(model, x_train, y_train, x_test, y_test, result_file)
