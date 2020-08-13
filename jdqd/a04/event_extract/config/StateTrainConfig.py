#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件状态模型训练需要的参数以就文件保存路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path
from jdqd.common.event_emm.model_utils import generate_trained_model_path


# 训练后模型保存路径
trained_model_dir = "event_extract/event_state_trained_model"
trained_model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_dir)
# 训练后模型名称
trained_model_name = "state_model.h5"
# 训练后模型路径
trained_model_path = generate_trained_model_path(trained_model_dir, trained_model_name)
# 训练集数据保存路径
train_data_path = "event_extract/data/event_state_data/raw_data/train_data.json"
train_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, train_data_path)
# 测试集数据保存路径
dev_data_path = "event_extract/data/event_state_data/raw_data/dev_data.json"
dev_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dev_data_path)
# 补充数据保存文件夹
supplement_data_dir = "event_extract/data/event_state_data/supplement"
supplement_data_dir = cat_path(appconf.ALGOR_PRETRAIN_ROOT, supplement_data_dir)
# 模型状态 下标 字典
state2id = {'happened': 0, 'happening': 1, 'possible': 2}
id2state = {0: 'happened', 1: 'happening', 2: 'possible'}
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