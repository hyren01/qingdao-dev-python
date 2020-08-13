#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件归并模型训练所有的参数和文件路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 批量大小
batch_size = 8
# 最大最小学习率
learning_rate = 5e-5
min_learning_rate = 1e-5
# 训练集路径
train_data_path = "event_search/train/data/datav06/train.txt"
train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, train_data_path)
# 验证集路径
dev_data_path = "event_search/train/data/datav06/dev.txt"
dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dev_data_path)
# 测试集路径
test_data_path = "event_search/train/data/datav06/test.txt"
test_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, test_data_path)
# 训练后的模型路径
trained_model_path = "event_search/trained_model/match_model.h5"
trained_model_path = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_path)
# 转化后的向量路径
vector_data_path = "event_search/train/data/vector_data"
vector_data_path = cat_path(appconf.ALGOR_MODULE_ROOT, vector_data_path)
# 存储转化样本的字典
vector_id_dict_dir = "event_search/train/data/"
vector_id_dict_dir = cat_path(appconf.ALGOR_MODULE_ROOT, vector_id_dict_dir)
