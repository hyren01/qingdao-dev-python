#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月11
from feedwork.utils.FileHelper import cat_path
import feedwork.AppinfoConf as appconf

# 模型类型
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
