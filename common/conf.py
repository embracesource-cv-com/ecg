# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/3/4 
@file: conf.py
@description:
"""
data_path = '/home/dataset/medical/preliminary'
label_path = '/home/dataset/medical/preliminary/reference.txt'
output_dir = '/home/model_output/medical/ecg/preliminary'

data_suffix = 'mat'
num_class = 2
train_ratio = 0.5
seed = 2019

wavelet_func = 'db6'
wavelet_level = 5
sample_rate = 500
feature_type = ['wavelet','heart_rate']  #'wavelet','heart_rate','distance','amplitude'

conv_kernel_size = 32
seq_len = 5000
num_lead = 12
batch_size = 60
dropout_rate = 0.5
weight_decay = 0.0005
lr = 0.001
patience = 50
steps_per_epoch =(train_ratio*600)//batch_size
print('steps_per_epoch:',steps_per_epoch)
epochs = 500
one_hot = False

gpu_index = "1"
ensemble = False
ensemble_mode = 'prob_sum'  # label_vote or prob_sum or prob_max
num_model = 5

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