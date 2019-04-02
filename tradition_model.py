# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/1 
@file: tradition_model.py
@description:
"""

from sklearn import svm
import time
from model_utils import evaluate, save_result
import pickle
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import conf


def do_svm(x_train, y_train, x_test, y_test, save_model, result_file):
    # kernel = ['Linear','poly','rbf']
    kernel = ['Linear', 'rbf']
    clf = []
    clf.append(svm.LinearSVC())
    # clf.append(svm.SVC(kernel='poly'))
    clf.append(svm.SVC(kernel='rbf'))

    trainAccu = []
    valiAccu = []
    for i in range(len(clf)):
        start_time = time.time()
        clf[i].fit(x_train, y_train)
        y_train_pred = clf[i].predict(x_train)
        y_test_pred = clf[i].predict(x_test)
        trainAccu.append(metrics.accuracy_score(y_train, y_train_pred))
        valiAccu.append(metrics.accuracy_score(y_test, y_test_pred))
        print('---' + kernel[i] + '_kernel---' + '\n' + "--- %s min ---" % ((time.time() - start_time) / 60))

    # save best model
    m = max(valiAccu)
    maxLoc = [i for i, j in enumerate(valiAccu) if j == m]
    i = maxLoc[0]
    model = clf[i]
    if save_model:
        pickle.dump(model, open("svm.pickle.dat", "wb"))

    # save model performace
    evaluate(model, x_train, y_train, x_test, y_test, result_file)


def do_knn(x_train, y_train, x_test, y_test, save_model, result_file):
    model = KNeighborsClassifier(n_neighbors=conf.num_class)
    model.fit(x_train, y_train)
    if save_model:
        pickle.dump(model, open("knn.pickle.dat", "wb"))

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
        pickle.dump(model, open("doXgb.pickle.dat", "wb"))
    # save model performace
    evaluate(model, x_train, y_train, x_test, y_test, result_file)

