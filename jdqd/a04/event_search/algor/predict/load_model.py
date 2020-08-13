#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
调用bert原始模型进行向量转化
加载匹配模型计算句子向量的相似度
"""
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Concatenate, Input
from bert4keras.models import build_transformer_model
from feedwork.utils import logger
import jdqd.a04.event_search.config.PredictConfig as pre_config
import jdqd.common.event_emm.BertConfig as bert_config

GRAPH_1 = tf.Graph()  # 加载到Session 1的graph
GRAPH_2 = tf.Graph()  # 加载到Session 2的graph

SESS_1 = tf.Session(graph=GRAPH_1)  # Session1
SESS_2 = tf.Session(graph=GRAPH_2)  # Session2


def load_bert_model():
    """
    加载bert模型，用于向量生成
    :return: bert_model(模型对象)
    """
    logger.info("开始加载bert模型。。。")
    # 在新的图上构建模型
    with SESS_1.as_default():
        with SESS_1.graph.as_default():
            # 建立bert模型，加载权重
            bert_model = build_transformer_model(bert_config.config_path, bert_config.checkpoint_path, return_keras_model=True)
    logger.info("bert模型加载完成!")

    return bert_model


def generate_vec(tokenizer, bert_model, sentence: str):
    """
    传入bert分词器、bert模型对象、待向量化的中文句子，对句子进行编码，然后使用bert进行向量化。
    :param tokenizer:分字器
    :param bert_model:模型对象
    :param sentence:(str)待向量化的句子
    :return:x(ndarray)句子向量(768,)
    :raise: TypeError
    """
    if not isinstance(sentence, str):
        logger.error("待向量化的句子格式错误，应该使用字符串格式！")
        raise TypeError

    # 句子编码, 按照最大长度切分
    token_ids, segment_ids = tokenizer.encode(first_text=sentence[:pre_config.maxlen])
    with SESS_1.as_default():
        with SESS_1.graph.as_default():
            # 句子向量
            x = bert_model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]

    return x


def load_match_model():
    """
    加载匹配模型，返回模型对象
    :return: 匹配模型对象
    """
    with SESS_2.as_default():
        with SESS_2.graph.as_default():
            # 句子向量
            x1_in = Input(shape=(768,))
            x2_in = Input(shape=(768,))
            # 拼接融合
            t = Concatenate(axis=1)([x1_in, x2_in])
            # 全连接层
            t = Dense(768, activation='relu')(t)
            # 计算相似度
            output = Dense(2, activation='softmax')(t)
            match_model = Model([x1_in, x2_in], [output])

            logger.info("开始加载匹配模型。。。")
            # 加载模型
            match_model.load_weights(pre_config.match_model_path)
            logger.info("匹配模型加载完成！")

    return match_model


def vec_match(x1, x2, match_model):
    """
    传入两个句子向量以及匹配模型对象，计算两个向量的相似度。
    :param x1: (ndarray)句子向量(768,)
    :param x2: (ndarray)句子向量(768,)
    :param match_model: 匹配模型
    :return: pred[0][1]（float）相似度值
    """
    with SESS_2.as_default():
        with SESS_2.graph.as_default():
            # 计算两个向量的相似度[[0.1,0.9]]
            pred = match_model.predict([np.array([x1]), np.array([x2])])

    return pred[0][1]

