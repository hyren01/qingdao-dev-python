# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:06:01 2020

@author: 12894
"""
import os
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Input, Lambda, Dense
from keras.models import Model

from jdqd.common.relation_com.model_utils import get_bert_tokenizer
import \
jdqd.a04.relation.relation_extract.config.relation_extract_config as CONFIG
import tensorflow as tf

g0 = tf.Graph()
ss0 = tf.Session(graph=g0)

# 使用bert字典，构建tokenizer类
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)

def sens_tagging(sentence, index_first, index_second):
            '''
    关系事件对谓语正向标注
    :param sentence: 字符串句子
    :param index_first: 正向关系第一个谓语下标(list)
    :param index_second: 正向关系第二个谓语下标(list)
    return 标注后的句子
    '''
    if index_first[1] <= index_second[0]:
        sentence = sentence[:index_first[0]] + '$' + sentence[index_first[0]:index_first[1]] + '$' + \
            sentence[index_first[1]:index_second[0]] + '#' + sentence[index_second[0]:index_second[1]] + '#' + \
            sentence[index_second[1]:]
    else:
        sentence = sentence[:index_second[0]] + '#' + sentence[index_second[0]:index_second[1]] + '#' +\
        sentence[index_second[1]:index_first[0]] + '$' + sentence[index_first[0]:index_first[1]] + '$' + \
        sentence[index_first[1]:]
    return  sentence


def load_model(relation_category, is_bi_directional):
    '''
    构建模型主体
    :param relation_category: 加载模型的类别
    :is_bi_direction: 是否双向关系
    return 模型对象
    '''
    # 搭建bert模型主体
    if is_bi_directional == 'T':
        n_class = 3
    else:
        n_class = 2
    with ss0.as_default():
        with ss0.graph.as_default():
            bert_model = load_trained_model_from_checkpoint(CONFIG.config_path,
                                                            CONFIG.checkpoint_path,
                                                            seq_len=None)  # 加载预训练模型
            for l in bert_model.layers:
                l.trainable = True

            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
            # x = Dense(256, activation='relu')(x)
            p = Dense(n_class, activation='softmax')(x)
            model = Model([x1_in, x2_in], p)
            model.load_weights(f'{CONFIG.relation_extract_model_path}/{relation_category}_relation_extract.h5')
    return model

model_lists = os.listdir(CONFIG.relation_extract_model_path)
for model_dir in model_lists:
    if model_dir.endswith(".h5"):
        relation_category = model_dir.split('_')[0]
        if relation_category == 'causality':
            causality_model = load_model(relation_category, 'T')
        elif relation_category == 'assumption':
            assumption_model = load_model(relation_category, 'T')
        elif relation_category == 'contrast':
            contrast_model = load_model(relation_category, 'T')
        elif relation_category == 'further':
            further_model = load_model(relation_category, 'T')
        elif relation_category == 'parallel':
            parallel_model = load_model(relation_category, 'F')
            

def class_pre(sentence, index_first, index_second, relation_category_model):
    '''
    输入事件对，得到关系结果
    :param sentence: 带预测句子
    :param index_first: 关系对第一个谓语的下标(list)
    :param index_second: 关系对第二个谓语的下标(list)
    return 关系判断
    '''
    test = sens_tagging(sentence, index_first, index_second)
    t1, t1_ = TOKENIZER.encode(first=test)
    T1, T1_ = np.array([t1]), np.array([t1_])
    with ss0.as_default():
        with ss0.graph.as_default():
            _prob = relation_category_model.predict([T1, T1_])
    prob = np.argmax(_prob)
    return prob

if __name__ == '__main__':
    sentence = '个别成员继续阻挠“临时安排”的做法，既缺乏世贸规则依据，也将进一步损害多边贸易体制。'
    index_first = [33, 35]
    index_second = [6, 8]
    prob1 = class_pre(sentence, index_first, index_second, parallel_model)
#     prob2 = class_pre(sentence, index_first, index_second, assumption_model)
    print(prob1)
#     print(prob2)
