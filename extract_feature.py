# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/22 
@file: extract_feature.py
@description:
"""
import numpy as np
import utils
import conf
from tqdm import tqdm
import wfdb
from wfdb import processing
import pywt
from detect_points import detect_points_in_all
import multiprocessing as mp
import time


def reject_outliers(data, m=2.):
    '''
    :param data: 1-D array
    :param m: remove a value if it is m times away from median
    :return: 1-D array with outliers removed
    '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def signal_heart_rate(signal):
    lead_mean_rates = []
    for sig in signal:
        r_locs = processing.xqrs_detect(sig=sig, fs=conf.sample_rate, verbose=False)
        lead_mean_rate = 60 / (np.nanmean(np.diff(r_locs)) / conf.sample_rate)
        lead_mean_rates.append(lead_mean_rate)
    lead_mean_rates = np.array([x for x in lead_mean_rates if str(x) != 'nan'])
    lead_mean_rates = reject_outliers(lead_mean_rates, m=2.)
    signal_mean_rate = np.mean(lead_mean_rates)
    return signal_mean_rate


def cal_heart_rate(signals):
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(signal_heart_rate, args=(signals[x],)) for x in range(len(signals))]
    heart_rate_feature = np.array([result.get() for result in results])
    heart_rate_feature = np.expand_dims(heart_rate_feature, axis=1)
    return heart_rate_feature


def cal_amp_feature(p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end, s_end, t_end):
    '''
    幅值特征, amp for amplitude，不同点之间的垂直距离，幅值加减顺序使正常心拍的特征的幅值都>0
    p_peak, q_peak, r_peak, s_peak, t_peak,q_begin, p_begin, t_begin, p_end, s_end, t_end: 2-d,array,(num_beats,[x,y])
    return: 2-D array，（num_beats,num_amplitude_feature)
    '''
    # 幅值特征, amp for amplititude，使正常心拍的振幅都>0
    p_q_amp = p_peak[:, 1] - q_peak[:, 1]  # P峰--Q峰
    r_q_amp = r_peak[:, 1] - q_peak[:, 1]  # Q峰--R峰
    r_s_amp = r_peak[:, 1] - s_peak[:, 1]  # R峰--S峰
    t_s = t_peak[:, 1] - s_peak[:, 1]  # S峰--T峰

    pb_b_amp = p_peak[:, 1] - p_begin[:, 1]  # P开始--P峰，无P波，室早
    pe_qb_amp = p_end[:, 1] - q_begin[:, 1]  # P结束--Q开始（PR段水平）
    qb_q_amp = q_begin[:, 1] - q_peak[:, 1]  # Q开始--Q峰
    s_se_amp = s_end[:, 1] - s_peak[:, 1]  # S峰--S结束
    se_tb_amp = t_begin[:, 1] - s_end[:, 1]  # S结束--T开始(ST段水平 )
    t_tb_amp = t_peak[:, 1] - t_begin[:, 1]  # T开始--T峰，T波方向与主波相反，室早

    amp_feature = [p_q_amp, r_q_amp, r_s_amp, t_s,
                   pb_b_amp, pe_qb_amp, qb_q_amp, s_se_amp, se_tb_amp, t_tb_amp]
    amp_feature = np.array(amp_feature).transpose()
    print('amp_feature shape:', amp_feature.shape)
    return amp_feature


def cal_dis_feature(p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end, s_end, t_end):
    '''
    距离特征，不同点之间的水平距离
    p_peak, q_peak, r_peak, s_peak, t_peak,q_begin, p_begin, t_begin, p_end, s_end, t_end: 2-d,array,(num_beats,[x,y])
    return: 2-D array，（num_beats,num_distance_feature)
    '''
    pb_qb_dis = q_begin[:, 0] - p_begin[:, 0]  # P开始--Q开始，PR间期，>0.12s,房性早搏
    qb_tb_dis = t_begin[:, 0] - q_begin[:, 0]  # Q开始--T开始，QT间期，变宽，室性早搏
    qb_r_se_dis = s_end[:, 0] - q_begin[:, 0]  # Q开始--S结束，QRS间期，>0.12s，束支传导阻滞

    pb_pe_dis = p_end[:, 0] - p_begin[:, 0]  # P开始--P结束，P波间期
    tb_te_dis = t_end[:, 0] - t_begin[:, 0]  # T开始--T结束，T波间期
    se_te_dis = t_end[:, 0] - s_end[:, 0]  # S结束--T结束，ST间期

    r_q_dis = r_peak[:, 0] - q_peak[:, 0]  # Q峰--R峰
    r_s_dis = s_peak[:, 0] - r_peak[:, 0]  # R峰--S峰
    r_p_dis = r_peak[:, 0] - p_peak[:, 0]  # P峰--R峰
    r_t_dis = t_peak[:, 0] - r_peak[:, 0]  # R峰--T峰

    pb_te_dis = t_end[:, 0] - p_begin[:, 0]  # P开始--T结束，整个心拍距离
    dis_feature = [pb_qb_dis, qb_tb_dis, qb_r_se_dis,
                   pb_pe_dis, tb_te_dis, se_te_dis,
                   r_q_dis, r_s_dis, r_p_dis, r_t_dis, pb_te_dis]

    dis_feature = np.round(np.array(dis_feature) / conf.sample_rate, 4)
    dis_feature = np.array(dis_feature).transpose()
    print('dis_feature shape:', dis_feature.shape)
    return dis_feature


def cal_wavelet_feature(signals, func=conf.wavelet_func, level=conf.wavelet_level):
    coeffs = pywt.wavedec(signals, wavelet=func, level=level)
    wavelet_feature = coeffs[0]
    wavelet_feature = wavelet_feature.reshape(len(signals), -1)
    print('wavelet_feature shape:', wavelet_feature.shape)
    return wavelet_feature


def get_all_feature(signals):
    feature_types = []
    if 'wavelet' in conf.feature_type:
        print('小波特征计算中...')
        wavelet_feature = cal_wavelet_feature(signals)
        feature_types.append(wavelet_feature)

    if 'heart_rate' in conf.feature_type:
        print('心率特征计算中...')
        heart_rate_feature = cal_heart_rate(signals)
        feature_types.append(heart_rate_feature)

    if 'distance' in conf.feature_type or 'amplitude' in conf.feature_type:
        print('心拍特征点定位中...')
        p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end, s_end, t_end = detect_points_in_all()
        if 'distance' in conf.feature_type:
            dis_feature = cal_dis_feature(p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end,
                                          s_end, t_end)
            feature_types.append(dis_feature)
        if 'amplitude' in conf.feature_type:
            amp_feature = cal_amp_feature(p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end,
                                          s_end, t_end)
            feature_types.append(amp_feature)

    all_feature = np.concatenate(feature_types, axis=1)
    print('all_feature shape:', all_feature.shape)
    return all_feature
