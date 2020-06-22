# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:24:45 2020

@author: 12894
"""
import codecs
import os
import time
import numpy as np
import tensorflow as tf
from keras_bert import Tokenizer
import keras.backend as K
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path


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

def seq_padding(seq, padding=0):
    """
    对传入的序列进行填充
    :param seq: (list)传入的待填充的序列
    :param padding: 填充字符
    :return: 返回填充后的数组数据
    """
    length = [len(x) for x in seq]
    max_len = max(length)
    return np.array([
        np.concatenate([x, [padding] * (max_len - len(x))]) if len(x) < max_len else x for x in seq
    ], dtype=np.float32)
    
    
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