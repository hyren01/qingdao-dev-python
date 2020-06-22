#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
基于向量化方法事件归并预测模块所有的参数
"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# bert所能编码的最大长度
maxlen = 512
# 匹配模型路径
match_model_path = "event_search/predict/model/match_model.h5"
match_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, match_model_path)
# 向量保存的文件夹
vec_data_dir = "event_search/predict/vec_data"
vec_data_dir = cat_path(appconf.ALGOR_MODULE_ROOT, vec_data_dir)
# cameo:事件id 字典文件
cameo2id_path = "event_search/predict/cameo2id.json"
cameo2id_path = cat_path(appconf.ALGOR_MODULE_ROOT, cameo2id_path)
