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
maxlen = 512
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5
# 训练数据集路径
train_data_path = "event_match/data/second_train/train.txt"
train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, train_data_path)
dev_data_path = "event_match/data/second_train/dev.txt"
dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dev_data_path)
test_data_path = "event_match/data/second_train/test.txt"
test_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, test_data_path)
# 训练后模型保存路径
trained_model_path = "event_match/model/trained_model/match_model.h5"
trained_model_path = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_path)
