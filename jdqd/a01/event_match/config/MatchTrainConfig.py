#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
匹配模型训练需要的所有参数和模型路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 训练批次大小
batch_size = 8
# 循环
epoch = 5
# dropout
drop_out_rate = 0.1
# 字符串最大长度
maxlen = 256
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5
# 首次训练数据集路径
first_train_data_path = "event_match/data/first_train/train.txt"
first_train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, first_train_data_path)
first_dev_data_path = "event_match/data/first_train/dev.txt"
first_dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, first_dev_data_path)
first_test_data_path = "event_match/data/first_train/test.txt"
first_test_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, first_test_data_path)
# 二次训练数据集路径
second_train_data_path = "event_match/data/second_train/train.txt"
second_train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, second_train_data_path)
second_dev_data_path = "event_match/data/second_train/dev.txt"
second_dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, second_dev_data_path)
second_test_data_path = "event_match/data/second_train/test.txt"
second_test_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, second_test_data_path)
# 首次训练后模型保存路径
first_trained_model_path = "event_match/model/trained_model/first_match_model.h5"
first_trained_model_path = cat_path(appconf.ALGOR_MODULE_ROOT, first_trained_model_path)
# 二次训练后模型保存路径
second_trained_model_path = "event_match/model/trained_model/match_model.h5"
second_trained_model_path = cat_path(appconf.ALGOR_MODULE_ROOT, second_trained_model_path)
