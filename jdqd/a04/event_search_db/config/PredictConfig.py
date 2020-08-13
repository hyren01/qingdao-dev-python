#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
基于向量化方法事件归并预测模块所有的参数
"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 夹角余弦值的阈值
cos_thread = 0.8
# bert所能编码的最大长度
maxlen = 512
# 匹配模型路径
match_model_path = "event_search/predict/model/match_model.h5"
match_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, match_model_path)

# 数据库相关配置
db_host = "139.9.126.19"
db_port = "31001"
db_name = "ebmdb2"
db_user = "jdqd"
db_passwd = "jdqd"
