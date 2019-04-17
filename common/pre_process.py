# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/2 
@file: pre_process.py
@description:
"""

import pywt
import numpy as np


def rm_noise(signal, wave_name='bior2.6', level=8):
    """
    小波去除高频噪声和基线漂移
    :param signal: 1-D array, 信号
    :param wave_name: str, eg.'bior2.6',小波函数名
    :param level: int，eg.8 小波分解阶数
    :return: 1-D array，去除高频噪声和基线漂移的信号
    """
    # 小波分解
    a_n = []
    d_n = []
    for i in range(level):
        c_a, c_d = pywt.dwt(signal, wave_name)
        a_n.append(c_a)
        d_n.append(c_d)
        signal = c_a
    a_n[level - 1] = np.zeros(a_n[level - 1].shape)  # 去除基线漂移
    d_n[0] = np.zeros(d_n[0].shape)  # 去除高频噪声
    d_n[1] = np.zeros(d_n[1].shape)  # 去除高频噪声
    # 小波重构
    wave_upper = a_n
    wave_lower = pywt.idwt(wave_upper[level-1], d_n[level-1], wave_name)
    wave_upper = wave_lower
    for i in reversed(range(level-1)):
        wave_lower = pywt.idwt(wave_upper[:len(d_n[i])], d_n[i], wave_name)
        wave_upper = wave_lower
    return wave_upper


def rm_noise_2d(signals, wave_name='bior2.6', level=8):
    results = np.empty_like(signals)
    for i in range(len(signals)):
        result = rm_noise(signals[i], wave_name, level)
        results[i] = result
    return results


def rm_noise_3d(signals, wave_name='bior2.6', level=8):
    results = np.empty_like(signals)
    for i in range(len(signals)):
        result = rm_noise_2d(signals[i], wave_name, level)
        results[i] = result
    return results
