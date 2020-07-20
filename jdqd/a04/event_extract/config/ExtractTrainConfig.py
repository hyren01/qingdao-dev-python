#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件抽取模型训练需要的所有参数以及文件路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 模型相关参数
model_type = "bert"
# 模型参数
config_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
config_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, config_path)
# 初始化模型路径
checkpoint_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, checkpoint_path)
# 模型字典路径
dict_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
dict_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dict_path)
# 训练后模型保存路径
trained_model_dir = "event_extract/event_extract_trained_model"
trained_model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_dir)
# 训练后模型名称
trained_model_name = "extract_model.h5"
# 训练集数据保存路径
train_data_path = "event_extract/data/event_extract_data/raw_data/train_data.json"
train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, train_data_path)
# 测试集数据保存路径
dev_data_path = "event_extract/data/event_extract_data/raw_data/dev_data.json"
dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dev_data_path)
# 补充数据保存文件夹
supplement_data_dir = "event_extract/data/event_extract_data/supplement"
supplement_data_dir = cat_path(appconf.ALGOR_PRETRAIN_ROOT, supplement_data_dir)
# 训练集测试集划分比例
ratio = 8
# 训练批次大小
batch_size = 8
# 循环
epoch = 100
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5
# 字符串最大长度
maxlen = 160
# pretrain_model
pretrain_model = "roberta"