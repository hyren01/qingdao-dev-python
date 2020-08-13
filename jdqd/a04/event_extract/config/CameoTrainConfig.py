#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件cameo模型训练需要的所有参数

"""
import feedwork.AppinfoConf as appconf
from jdqd.common.event_emm.model_utils import generate_trained_model_path
from feedwork.utils.FileHelper import cat_path


# 训练后模型保存路径
trained_model_dir = "event_cameo_trained_model"
trained_model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_dir)
# 训练后模型名称
trained_model_name = "cameo_model.h5"
# 训练后模型保存路径
trained_model_path = generate_trained_model_path(trained_model_dir, trained_model_name)
# 训练集数据保存路径
train_data_path = "event_extract/data/event_cameo_data/train_data.json"
train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, train_data_path)
# 测试集数据保存路径
dev_data_path = "event_extract/data/event_cameo_data/dev_data.json"
dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dev_data_path)
# 标签字典
label2id_path = "event_extract/data/event_cameo_data/label2id.json"
label2id_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, label2id_path)
id2label_path = "event_extract/data/event_cameo_data/id2label.json"
id2label_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, id2label_path)
# 训练批次大小
batch_size = 8
# 循环
epoch = 100
# dropout
drop_out_rate = 0.3
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5
# 字符串最大长度
maxlen = 160
# pretrain_model
pretrain_model = "roberta"