#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
存放模型数据处理需要的所有公用的方法，分字器加载、向量融合以及训练模型临时文件夹生成等
"""
import codecs
import os
import time
import keras
import numpy as np
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import K
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
from jdqd.common.event_emm.BertConfig import dict_path


def get_token_dict(dict_path: str):
    """
    传入bert字典路径，按行读取文件内容，返回字典
    :param dict_path: (str)字典文件路径
    :return: token_dict(dict)模型字典
    """
    if not os.path.exists(dict_path) or os.path.isdir(dict_path) or not dict_path.endswith(".txt"):
        logger.error(f"{dict_path} bert 字典文件损坏！")
        raise ValueError

    token_dict = {}
    # 加载模型对应的字典
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        # 遍历读取文件内容的每一行
        for line in reader:
            token = line.strip()
            # 将字典的长度作为字符的id
            token_dict[token] = len(token_dict)

    return token_dict


class OurTokenizer(Tokenizer):
    """
    改写bert自带的分字类，处理空白符和不识别字符的分字问题。
    """

    def _tokenize(self, text: str):
        """
        改写Tokenizer类中的分字函数，对传入的字符串进行分字，
        对空白符以及未知字符分别使用[unused1] 和[UNK] 占位。
        :param text: (str)字符串
        :return:token_list(list)分字列表
        """
        token_list = []
        for c in text:
            if c in self._token_dict:
                token_list.append(c)
            # 对于空白符使用bert中未训练的[unused1]占位
            elif self._is_space(c):
                token_list.append('[unused1]')
            # 剩余的未知字符使用[UNK]占位
            else:
                token_list.append('[UNK]')

        return token_list


def get_bert_tokenizer(dict_path: str):
    """
    传入字典路径，读取字典内容，并使用字典初始化分字器，返回分字器对象
    :param dict_path: (str)字典路径
    :return: tokenizer(object)分字器
    :raise: ValueError 如果字典文件不存在或者字典文件不是.txt文件
    """
    if not os.path.exists(dict_path) or os.path.isdir(dict_path) or not dict_path.endswith(".txt"):
        logger.error(f"{dict_path} bert 字典文件损坏！")
        raise ValueError

    # 加载bert字典
    token_dict = get_token_dict(dict_path)
    # 构建tokenizer类
    tokenizer = OurTokenizer(token_dict)

    return tokenizer


# 加载bert分字器
TOKENIZER = get_bert_tokenizer(dict_path)


def seq_padding(seq: list, padding=0):
    """
    对传入多维列表使用填充单元padding进行填充，使得序列长度统一等于最大序列长度
    :param seq: (list)待填充的序列
    :param padding: 填充单元
    :return: 返回填充后的数组数据
    """
    # 统计列表中所有序列的长度
    length = [len(x) for x in seq]
    # 获取最大序列长度
    max_len = max(length)
    # 遍历整个列表，对每个序列使用填充单元padding进行填充，直到达到最大序列长度，并保存成ndarray
    return np.array([
        np.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in seq
    ], dtype=np.float32)


def seq_gather(x: list):
    """
    传入从传入的列表x中获取句子张量seq和下标idxs
    seq是[batch_size, seq_len, vector_size]的形状，
    idxs是[batch_size, 1]的形状
    在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[batch_size, s_size]的向量。
    :param x: [seq, idxs] seq 原始序列的张量，idxs需要拆分的向量下标
    :return: 收集出来的字向量
    """
    # 获取句子张量以及字下标张量 idx = [[4],[9],[8],[11],[23],[45],[60],[30]]
    seq, idxs = x
    # 将下标数据类型转化为整型
    idxs = K.cast(idxs, 'int32')
    # 使用keras方法构造0-batch_size的张量[0,1,2,3,4,5,6,7]
    batch_idxs = K.arange(0, K.shape(seq)[0])
    # 在batch_idxs中扩充维度1，为的是与idx进行拼接后到seq中取切分向量[[0],[1],[2],[3],[4],[5],[6],[7]]
    batch_idxs = K.expand_dims(batch_idxs, 1)
    # 拼接idxs与batch_idx [[0,4],[1,9],[2,8],[3,11],[4,23],[5,45],[6,60],[7,30]]
    idxs = K.concatenate([batch_idxs, idxs], 1)
    # 对应idxs下标将seq中对应位置的向量收集出来
    return tf.gather_nd(seq, idxs)


def list_find(list1, list2):
    """
    在序列list1中寻找子串list2,如果找到，返回第一个下标；
    如果找不到，返回-1
    :param list1: (list, str)
    :param list2: (list, str)
    :return: i(int)起始下标， -1 未找到
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def generate_trained_model_path(trained_model_dir: str, trained_model_name: str):
    """
    传入模型保存文件夹，和训练后模型名称，按照日期生成模型保存文件夹，并生成模型保存路径，返回模型保存路径
    :param trained_model_dir: (str)模型保存文件夹
    :param trained_model_name: (str)模型名称
    :return: trained_model_path(str)模型保存路径
    :raise:ValueError 文件夹不存在或者模型命名不以.h5结尾
    """
    # 判断保存训练后模型的文主文件夹是否存在，不存在则报错
    if not os.path.exists(trained_model_dir) or not os.path.isdir(trained_model_dir):
        logger.error(f"{trained_model_dir} is wrong!")
        raise ValueError
    # 模型不以.h5结尾则报错
    if not trained_model_name.endswith(".h5"):
        logger.error(f"{trained_model_name} must end with .h5")
        raise ValueError
    # 按照当前时间年月日生成文件夹，构建训练后模型的保存位置
    trained_model_dir = cat_path(trained_model_dir, time.strftime('%Y-%m-%d', time.localtime(time.time())))
    trained_model_path = cat_path(trained_model_dir, trained_model_name)
    # 如果文件夹不存在则生成
    if not os.path.exists(trained_model_dir):
        os.mkdir(trained_model_dir)

    return trained_model_path


def search_layer(inputs, name, exclude=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude is None:
        exclude = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer


def adversarial_training(model, embedding_name, epsilon=0.1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs


    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    embedding_layer = None

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    model.train_function = train_function  # 覆盖原训练函数