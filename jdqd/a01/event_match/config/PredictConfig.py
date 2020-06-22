#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件提取中匹配模型预测模块需要的所有参数以及文件路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 模型批量大小
batch_size = 1
# dropout
drop_out_rate = 0.1
# 字符串最大长度
maxlen = 256
# 事件类别模型路径
match_model_path = "event_match/model/trained_model/match_model.h5"
match_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, match_model_path)
# 事件列表地址
allevent_path = "event_match/data/allevent"
allevent_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, allevent_path)
