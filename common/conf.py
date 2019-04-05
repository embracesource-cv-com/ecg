# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/4
@file: conf.py
@description:
"""
import math

data_path = '/home/dataset/medical/preliminary_old'
label_path = '/home/dataset/medical/preliminary_old/reference.txt'
output_dir = '/home/model_output/medical/ecg/preliminary_old'
data_suffix = 'mat'
sample_rate = 500
num_class = 2
num_samples = 600
seq_len = 5000
num_lead = 12

train_ratio = 0.8
batch_size = 120
steps_per_epoch = math.floor(train_ratio*num_samples)//batch_size  # 最好保证整除，不然会有部分样本没参加训练
epochs = 100
conv_kernel_size = 32
dropout_rate = 0.5
weight_decay = 0.005
lr = 0.001
patience = 50

ensemble = True
ensemble_mode = 'prob_sum'  # label_vote or prob_sum or prob_max
num_model = 5

seed = 2019
gpu_index = "1"
one_hot = False

wavelet_func = 'db6'
wavelet_level = 5
feature_type = ['wavelet','heart_rate']  #'wavelet','heart_rate','distance','amplitude'

conv_kernel_size = 32
seq_len = 5000
num_lead = 12

dropout_rate = 0.5
weight_decay = 0.005
lr = 0.001
patience = 50

continue_training = False
weights_to_transfer = '/home/model_output/medical/ecg/weights.hdf5'
use_tradition_feature = False

#feature_type = ['wavelet']
#func_path = 'C:\\Users\\lidan\\OneDrive\\A_markdown\\public\\healthcare\\201902_心电图\\code\\201903_traditional_ecg'
#func_path = '/home/github/py_data_mining/medical/tradition_ecg'

'''
for name in dir():
    if not name.startswith('__'):
        myvalue = eval(name)
        print (name, " ", type(myvalue), "  ", myvalue)
'''