# -*- coding:utf-8 -*-
"""
   File Name：     detect_points.py
   Description :   批量提取心电节拍中的特征波点、波段
   Author :        royce.mao
   date：          2019/03/25

"""
import numpy as np
import common.utils as utils
from tqdm import tqdm


def beat_slicing():
    """
    数据加载与单个波形的切分
    :return: 
    """
    file_names = utils.get_file_name()
    all_x, all_y = utils.load_data(file_names)
    points = all_x.shape[1]
    return all_x, all_y, points


def QRS_points_detection(RA_upper, points):
    """
    # 检测单个心电信号的QRS波段中的波峰与波谷
    # 逻辑：局部极值与索引关系
    :param RA_upper: 
    :param points: 
    :return: 
    """
    pddw = []
    nddw = []
    # 原信号图中大于0的数值保持原值，小于0的数值全部置0
    posw = np.array([RA_upper[i] if RA_upper[i] > 0 else 0 for i in range(points)])
    # 计算所有斜率>0部分信号点的索引
    pdw = np.where(posw[:-1] - posw[1:] < 0)
    # 根据局部斜率变化情况，找出所有正极大值点的索引
    for pos_ids in pdw[0]:
        if pos_ids + 1 not in pdw[0]:
            pddw.append(pos_ids + 1)
    # 原信号图中小于0的数值保持原值，大于0的数值全部置0
    negw = np.array([RA_upper[i] if RA_upper[i] < 0 else 0 for i in range(points)])
    # 判断是否全是0
    # print(np.array([RA_upper[i] for i in range(points)]))
    # if len(np.where(negw == 0)[0]) == len(negw):
    #     negw = np.array([RA_upper[i] if RA_upper[i] < 1.2 else 0 for i in range(points)])
    # 计算所有斜率<0部分信号点的索引
    pdw = np.where(negw[:-1] - negw[1:] > 0)
    # 根据局部斜率变化情况，找出所有负极大值点的索引
    for neg_ids in pdw[0]:
        if neg_ids + 1 not in pdw[0]:
            nddw.append(neg_ids + 1)
    # 正、负极值索引list的合并，非索引下的取值置0，索引下的取值保持
    ddw = pddw + nddw
    swd_3 = np.zeros((1, points))
    for inds in ddw:
        swd_3[..., inds] = RA_upper[inds]
    # 找到R波峰与其相邻的Q、S波谷
    # swd_3_pos = np.zeros((1, points))
    # swd_3_neg = np.zeros((1, points))
    R_ind = pddw[np.argmax(RA_upper[pddw])]
    for i, ind in enumerate(nddw):
        if ind > R_ind:
            Q_ind = nddw[i - 1]
            S_ind = ind
            break
    # swd_3_pos[0, R_ind] = RA_upper[R_ind]
    # swd_3_neg[0, [Q_ind, S_ind]] = RA_upper[[Q_ind, S_ind]]
    # max_inds = np.where(swd_3_pos[0] > 0)
    # min_inds = np.where(swd_3_neg[0] < 0)
    # 波峰、波谷点的坐标对输出
    x_R, y_R = R_ind, RA_upper[R_ind]
    x_Q, y_Q = Q_ind, RA_upper[Q_ind]
    x_S, y_S = S_ind, RA_upper[S_ind]
    # 返回R波峰的点坐标、Q、S两个波谷点的坐标
    return [x_R, y_R], [x_Q, y_Q], [x_S, y_S], [x_Q, x_S]


def QRS_band_detection(RA_upper, points, x_min):
    """
    # 根据寻找到的R、Q与S波点，界定QRS波段的起始位置与结束位置
    # 逻辑：变斜率阈值条件
    :param RA_upper: 
    :param points: 
    :param x_min: 
    :return: 
    """
    Q_start = []
    for left_Q in [x_min[0]]:
        slopes = []
        for left_inds in range(left_Q, -1, -1):
            slope_tmp = RA_upper[left_inds] - RA_upper[left_inds - 1]
            slopes.append(slope_tmp)
            slope_threshold = np.mean(np.array(slopes))
            # 变斜率阈值条件
            if slope_tmp * slope_threshold <= 0 or slope_tmp >= 0.5 * slope_threshold:
                Q_start.append(left_inds)
                break
    # # 所有S点的变斜率阈值确定QRS波段的结束点
    S_end = []
    for right_S in [x_min[1]]:
        slopes = []
        for right_inds in range(right_S, points):
            slope_tmp = RA_upper[right_inds + 1] - RA_upper[right_inds]
            slopes.append(slope_tmp)
            slope_threshold = np.mean(np.array(slopes))
            # 变斜率阈值条件
            if slope_tmp * slope_threshold <= 0 or slope_tmp <= 0.5 * slope_threshold:
                S_end.append(right_inds)
                break
    # # 边界点坐标对
    y_Q_start = RA_upper[np.array(Q_start)]
    y_S_end = RA_upper[np.array(S_end)]
    # 返回单个心拍QRS波段的左边界Q点坐标，与右边界S点坐标
    return [Q_start[0], y_Q_start[0]], [S_end[0], y_S_end[0]]


def TP_wave_detection(RA_upper, points, Q_start, S_end):
    """
    # 寻找T、P波峰，以及他们的起始点与终止点，即TB,TE,PB PE。（共6个点）
    # 逻辑：局部极值与差值绝对值的局域变换
    :param RA_upper: 
    :param points: 
    :param Q_start: 
    :param S_end: 
    :return: 
    """
    T_top = []
    P_top = []
    # 多个波形时可用RR波峰区间的自适应窗口
    RR_adaptive_window = np.stack([np.array([S_end[0]]), np.array([Q_start[0]])], axis=1)
    for RR_region in RR_adaptive_window:
        #   top = np.argmax(RA_upper[range(RR_region[0]+4900,RR_region[1]+4900)])+RR_region[0]
        T_top.append(np.argmax(RA_upper[range(RR_region[0], points)]) + RR_region[0])  # 切分左边为T波区域
        P_top.append(np.argmax(RA_upper[range(0, RR_region[1])]))  # 切分右边为P波区域
    # # 转numpy
    Q_start = np.array(Q_start)
    S_end = np.array(S_end)
    T_top = np.array(T_top)
    P_top = np.array(P_top)
    # # T波波段
    ### 左信号子段直线
    diff_left_T = []
    A = RA_upper[T_top] - RA_upper[int(S_end[0])]
    B = S_end[0] - T_top
    C = T_top * RA_upper[int(S_end[0])] - S_end[0] * RA_upper[T_top]  # 矩阵乘法运算
    for i in range(len(RR_adaptive_window)):
        try:
            diff_left_T.append(np.argmax(np.abs(RA_upper[range(RR_adaptive_window[i, 0], T_top[i])] +
                                                (C + A * range(RR_adaptive_window[i, 0], T_top[i])) / B)) +
                               RR_adaptive_window[i, 0])
        except Exception:
            diff_left_T.append(T_top[i])
    ### 右信号子段直线
    diff_right_T = []
    A = RA_upper[points - 1] - RA_upper[T_top]
    B = T_top - points + 1
    C = (points - 1) * RA_upper[T_top] - T_top * RA_upper[points - 1]  # 矩阵乘法运算
    for i in range(len(RR_adaptive_window)):
        try:
            diff_right_T.append(np.argmax(np.abs(RA_upper[range(T_top[i], points - 1)] +
                                                 (C + A * range(T_top[i], points - 1)) / B)) + T_top[i])
        except Exception:
            diff_right_T.append(T_top[i])
    # # P波波段（跟求T波波段的原理一样）
    ### 左信号子段直线
    diff_left_P = []
    A = RA_upper[P_top] - RA_upper[0]
    B = 0 - P_top
    C = P_top * RA_upper[0]  # 矩阵乘法运算
    for i in range(len(RR_adaptive_window)):
        try:
            diff_left_P.append(np.argmax(np.abs(RA_upper[range(0, P_top[i])] +
                                                (C + A * range(0, P_top[i])) / B)))
        except Exception:
            diff_left_P.append(0)
    ### 右信号子段直线
    diff_right_P = []
    A = RA_upper[int(Q_start[0])] - RA_upper[P_top]
    B = P_top - Q_start[0]
    C = Q_start[0] * RA_upper[P_top] - P_top * RA_upper[int(Q_start[0])]  # 矩阵乘法运算
    for i in range(len(RR_adaptive_window)):
        diff_right_P.append(np.argmax(np.abs(RA_upper[range(P_top[i], RR_adaptive_window[i, 1])] +
                                             (C + A * range(P_top[i], RR_adaptive_window[i, 1])) / B)) + P_top[i])
    # # 找到原信号图上P、T波有关的所有y值
    y_T_top = RA_upper[np.array(T_top)]
    y_P_top = RA_upper[np.array(P_top)]
    y_diff_left_T = RA_upper[np.array(diff_left_T)]
    y_diff_right_T = RA_upper[np.array(diff_right_T)]
    y_diff_left_P = RA_upper[np.array(diff_left_P)]
    y_diff_right_P = RA_upper[np.array(diff_right_P)]
    # 返回T、P波有关的2个波峰与4个边界点坐标
    return [T_top[0], y_T_top[0]], [P_top[0], y_P_top[0]], [diff_left_T[0], y_diff_left_T[0]], [diff_right_T[0],
                                                                                                y_diff_right_T[0]], [
               diff_left_P[0], y_diff_left_P[0]], [diff_right_P[0], y_diff_right_P[0]]


def achieve_all_points(RA_upper, points):
    """
    合并
    :param RA_upper: 
    :param points: 
    :return: 
    """
    r_peak, q_peak, s_peak, x_min = QRS_points_detection(RA_upper, points)
    q_begin, s_end = QRS_band_detection(RA_upper, points, x_min)
    t_peak, p_peak, t_begin, t_end, p_begin, p_end = TP_wave_detection(RA_upper, points, q_begin, s_end)
    return p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end, s_end, t_end


def detect_points_in_all():
    x, y, wave_len = beat_slicing()
    # 按照（p_peak,q_peak,r_peak,s_peak,t_peak,q_begin,p_begin,t_begin,p_end,s_end,t_end）的顺序，返回所有案例样本的特征点坐标
    ls_p_peak, ls_q_peak, ls_r_peak, ls_s_peak, \
    ls_t_peak, ls_q_begin, ls_p_begin, ls_t_begin, \
    ls_p_end, ls_s_end, ls_t_end = [], [], [], [], [], [], [], [], [], [], []
    # 根据需要遍历返回11个包含所有样本的特征点坐标numpy
    for sample in tqdm(x):
        try:
            # 非正常波形的特征点由于各种问题，这里就没考虑提取
            p_peak, q_peak, r_peak, s_peak, t_peak, q_begin, p_begin, t_begin, p_end, s_end, t_end = achieve_all_points(
                sample, wave_len)
            ls_p_peak.append(p_peak)
            ls_q_peak.append(q_peak)
            ls_r_peak.append(r_peak)
            ls_s_peak.append(s_peak)
            ls_t_peak.append(t_peak)
            ls_q_begin.append(q_begin)
            ls_p_begin.append(p_begin)
            ls_t_begin.append(t_begin)
            ls_p_end.append(p_end)
            ls_s_end.append(s_end)
            ls_t_end.append(t_end)
        except Exception:
            # 非正常波形的特征点坐标全部置零
            ls_p_peak.append([0, 0])
            ls_q_peak.append([0, 0])
            ls_r_peak.append([0, 0])
            ls_s_peak.append([0, 0])
            ls_t_peak.append([0, 0])
            ls_q_begin.append([0, 0])
            ls_p_begin.append([0, 0])
            ls_t_begin.append([0, 0])
            ls_p_end.append([0, 0])
            ls_s_end.append([0, 0])
            ls_t_end.append([0, 0])

    points = ls_p_peak, ls_q_peak, ls_r_peak, ls_s_peak, ls_t_peak, ls_q_begin, ls_p_begin, ls_t_begin, ls_p_end, ls_s_end, ls_t_end
    ls_p_peak, ls_q_peak, ls_r_peak, ls_s_peak, ls_t_peak, ls_q_begin, ls_p_begin, ls_t_begin, ls_p_end, ls_s_end, ls_t_end = [
        np.array(i) for i in points]
    return ls_p_peak, ls_q_peak, ls_r_peak, ls_s_peak, ls_t_peak, ls_q_begin, ls_p_begin, ls_t_begin, ls_p_end, ls_s_end, ls_t_end
