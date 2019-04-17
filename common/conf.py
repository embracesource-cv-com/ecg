# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/4
@file: conf.py
@description:
"""
import math


class Config(object):
    data_path = '/home/dataset/medical/preliminary'
    label_path = '/home/dataset/medical/preliminary/reference.txt'
    output_dir = '/home/model_output/medical/ecg/preliminary'
    data_suffix = 'mat'
    sample_rate = 500
    num_class = 2
    num_samples = 600
    seq_len = 5000
    seg_len = 250
    num_lead = 12
    lead_as_sample = False

    train_ratio = 0.6
    batch_size = 120
    epochs = 100
    conv_kernel_size = 16
    dropout_rate = 0.5
    weight_decay = 0.005
    lr = 0.001
    patience = 10
    model_1d = 'simple_net'  # ecg_resnet or mini_resnet or simple_net
    model_2d = 'resnet'  # resnet or simple_net

    ensemble = False
    ensemble_mode = 'prob_sum'  # label_vote or prob_sum or prob_max
    num_model = 5

    seed = 2019
    gpu_index = "1"
    one_hot = False
    if lead_as_sample:
        steps_per_epoch = math.floor(
            train_ratio * train_ratio * num_samples * num_lead) // batch_size  # 最好保证整除，不然会有部分样本没参加训练
    else:
        steps_per_epoch = math.floor(train_ratio * num_samples) // batch_size  # 最好保证整除，不然会有部分样本没参加训练

    wavelet_func = 'db6'
    wavelet_level = 5
    feature_type = ['wavelet', 'heart_rate']  # 'wavelet','heart_rate','distance','amplitude'

    continue_training = False
    # weights_to_transfer = '/home/model_output/medical/ecg/weights.hdf5'
    weights_to_transfer = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    use_tradition_feature = False


# 当前生效配置
current_config = Config()


# feature_type = ['wavelet']
# func_path = 'C:\\Users\\lidan\\OneDrive\\A_markdown\\public\\healthcare\\201902_心电图\\code\\201903_traditional_ecg'
# func_path = '/home/github/py_data_mining/medical/tradition_ecg'


for name in dir():
    if not name.startswith('__'):
        myvalue = eval(name)
        print(name, " ", type(myvalue), "  ", myvalue)
