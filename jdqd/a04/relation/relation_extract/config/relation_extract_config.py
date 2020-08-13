# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:30:26 2020

@author: 12894
"""

"""
关系抽取模型训练需要的所有参数以及文件路径

"""
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path

# 關系類型
relation_category = 'assumption'
# 是否bi_directional
is_bi_directional_dic = {'causality':'T', 'contrast':'T', 'assumption':'T', 'parallel':'F', 'further':'T'}
is_bi_directional = is_bi_directional_dic.get(relation_category)
# 关系抽取模型路径
relation_extract_model_path = "relation_extract/relation_extract_model"
relation_extract_model_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, relation_extract_model_path)
# 模型参数
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
config_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, config_path)
# 初始化模型路径
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, checkpoint_path)
# 模型字典路径
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
dict_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, dict_path)
# 训练后模型保存路径
trained_model_dir = "relation_extract_model"
trained_model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, trained_model_dir)
# 训练后模型名称
trained_model_name = f"{relation_category}_relation_extract.h5"
# 训练集数据保存路径
total_data_path = f"relation_extract/relation_extract_data/{relation_category}_train_data"
total_data_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, total_data_path)
# 训练后测试集预测结果
pred_path = f"relation_extract/test_pre/{relation_category}_test_pred.txt"
pred_path = cat_path(appconf.ALGOR_PRETRAIN_ROOT, pred_path)
# 循环
epoch = 50
#批次
batch_size = 10
# 字符串最大长度
maxlen = 160