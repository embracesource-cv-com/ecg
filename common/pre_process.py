# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/2 
@file: pre_process.py
@description:
"""

import pywt
import numpy as np


def rm_noise(signal, wave_name, level):
    '''
    小波去除高频噪声和基线漂移
    :param signal: 1-D array, 信号
    :param wave_name: str, eg.'bior2.6',小波函数名
    :param level: int，eg.8 小波分解阶数
    :return: 1-D array，去除高频噪声和基线漂移的信号
    '''
    # 小波分解
    An = []
    Dn = []
    for i in range(level):
        cA, cD = pywt.dwt(signal, wave_name)
        An.append(cA)
        Dn.append(cD)
        signal = cA
    An[level - 1] = np.zeros(An[level - 1].shape)  # 去除基线漂移
    Dn[0] = np.zeros(Dn[0].shape)  # 去除高频噪声
    Dn[1] = np.zeros(Dn[1].shape)  # 去除高频噪声
    # 小波重构
    for i in reversed(range(level)):
        if i == (level - 1):
            wave_upper = An
            wave_lower = pywt.idwt(wave_upper[i], Dn[i], wave_name)
        else:
            wave_lower = pywt.idwt(wave_upper[:len(Dn[i])], Dn[i], wave_name)
        wave_upper = wave_lower
    return wave_upper


def rm_noise_2d(signals, wave_name, level):
    results = np.empty_like(signals)
    for i in range(len(signals)):
        result=rm_noise(signals[i], wave_name, level)
        results[i]=result
    return results


def rm_noise_3d(signals, wave_name, level):
    results = np.empty_like(signals)
    for i in range(len(signals)):
        result = rm_noise_2d(signals[i], wave_name, level)
        results[i] = result
    return results