#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
传递事件抽取预测模块需要的所有参数
"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 模型类型
model_type = "bert"
# 字符串最大长度
maxlen = 160
# 事件抽取模型路径
event_extract_model_path = "event_extract/model/extract_model.h5"
event_extract_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, event_extract_model_path)
# 事件状态判断模型路径
event_state_model_path = "event_extract/model/state_model.h5"
event_state_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, event_state_model_path)
# 事件类别模型路径
event_cameo_model_path = "event_extract/model/cameo_model.h5"
event_cameo_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, event_cameo_model_path)
# bert模型参数json文件路径
bert_config_path = "chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
bert_config_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, bert_config_path)
# bert模型字典保存的路径
dict_path = "chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
dict_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dict_path)
# 小牛翻译的key
user_key = "b3d33c84a6291b89524e1a759064032a"
# 小牛翻译的网址
translate_url = "http://free.niutrans.com/NiuTransServer/translation"
# 小牛翻译的key
charge_user_key = "7bbd0ffbf54212e89b03d0aa120f9224"
# 小牛翻译的网址
charge_translate_url = "http://api.niutrans.com/NiuTransServer/translation"
# 事件类型字典文件
id2cameo_path = "event_extract/id2label.json"
id2cameo_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, id2cameo_path)
