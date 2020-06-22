#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月02
"""
存放事件匹配模块模型训练及预测数据生成类、加载数据和补充城市国家的方法
"""
import numpy as np
from jdqd.common.event_emm.data_utils import file_reader
from jdqd.common.event_emm.model_utils import seq_padding


def supplement(content: str):
    """
    传入待补充的字符串内容，然后对其中的城市进行国家补充，
    :param content: (str) 待补充的文本内容
    :return: 补充后的文本内容
    """
    # 将标志性城市后插入所属国家首尔--首尔（韩国）
    city_country = {'纽约': '美国', '北京': '中国',
                    '首尔': '韩国', '平壤': '朝鲜',
                    '东京': '日本', '莫斯科': '俄罗斯',
                    '白宫': '美国', '东仓里': '朝鲜'}
    for city in city_country:
        if city in content:
            content = content.replace(city, f"{city}{city_country[city]}")

    return content


def load_data(file_path: str):
    """
    传入文件路径，读取文件内容，返回模型需要的数据列表
    :param file_path: (str) 数据路径
    :return: data(list) 数据列表
    """
    content = file_reader(file_path)
    # 按行切分
    lines = [once for once in content.split("\n") if once]
    data = [(once.split("\t")[0], once.split("\t")[1], once.split("\t")[2]) for once in lines]

    return data


class DataGenerator(object,):
    """构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回
    """

    def __init__(self, data, tokenizer, max_length=128, batch_size=8):
        """
        接收分字器、最大长度、数据、批量大小
        :param data:(list)传入的数据
        :param tokenizer:分字器
        :param max_length:最大长度
        :param batch_size:批量大小
        """
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = max_length
        self.batch_size = batch_size
        # 计算数据步数
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        """
        :return:返回数据的步数
        """
        return self.steps

    def __iter__(self, random=False):
        """
        构造迭代方法
        :param random: bool 是否进行随机打乱
        :return:模型训练需要的数据
        """
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            # 单条数据拆分
            text1, text2, label = self.data[i]
            # 对文本对进行编码
            token_ids, segment_ids = self.tokenizer.encode(text1, text2, max_length=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(int(label))
            # 判断是否已经到达批量
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = seq_padding(batch_token_ids)
                batch_segment_ids = seq_padding(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        # 循环迭代输出数据
        while True:
            for d in self.__iter__(True):
                yield d
