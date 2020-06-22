#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
提供事件状态模型训练数据加载方法和数据生成类
"""
import os
import numpy as np
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
from jdqd.common.event_emm.model_utils import seq_padding
from jdqd.common.event_emm.data_utils import read_json, valid_file


def get_data(train_data_path: str, dev_data_path: str, supplement_data_dir: str):
    """
    传入训练集、验证集、补充数据路径，读取并解析json数据，将补充数据补充到训练集中，返回解析后的数据
    :param train_data_path:(str)训练集路径
    :param dev_data_path:(str)验证集路径
    :param supplement_data_dir:(str)补充数据保存路径
    :return:train_data(list), dev_data(list)
    """
    # 验证传入的路径是否存在问题
    if not valid_file(train_data_path):
        logger.error(f"训练数据{train_data_path} 路径错误！")
    elif not valid_file(dev_data_path):
        logger.error(f"验证数据{dev_data_path} 路径错误！")
    elif not os.path.exists(supplement_data_dir) or os.path.isfile(supplement_data_dir):
        logger.error(f"补充数据集{supplement_data_dir} 文件夹路径错误！")

    # 加载训练数据集
    train_data = read_json(train_data_path)
    # 加载验证集
    dev_data = read_json(dev_data_path)

    # 加载补充数据
    file_list = os.listdir(supplement_data_dir)
    supplement_data = []
    for file in file_list:
        supplement_data_path = cat_path(supplement_data_dir, file)
        supplement_data.extend(read_json(supplement_data_path))
    train_data.extend(supplement_data)

    return train_data, dev_data


class DataGenerator(object):
    """
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回。
    """

    def __init__(self, tokenizer, maxlen, data, state2id, batch_size=8, shuffle=True):
        """
        接收数据、批次大小，初始化实体参数。
        :param tokenizer: (object)分字器
        :param maxlen: (int)最大长度
        :param data: (list)数据
        :param state2id: (dict)状态下标转换字典
        :param batch_size: (int)批量大小
        :param shuffle: (bool)打乱
        """
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.data = data
        self.state2id = state2id
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 计算数据的步数
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        """
        :return: 返回该数据集的步数
        """
        return self.steps

    def __iter__(self):
        """
        构造生成器
        :return: 迭代返回批量数据
        """
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)  # 对传入的数据进行打乱

            # 初始化数据列表
            text_token_ids, text_segment_ids, trigger_start_index, trigger_end_index, labels = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d["sentence"][:self.maxlen]
                # 对句子进行编码
                text_token_id, text_segment_id = self.tokenizer.encode(first_text=text)
                for event in d["events"]:
                    # 动词下标
                    key = event["trigger"][0][1]
                    k1 = int(key[0]) + 1
                    k2 = int(key[1])
                    # 状态标签
                    y = self.state2id[event["state"]]
                    trigger_start_index.append([k1])
                    trigger_end_index.append([k2])
                    text_token_ids.append(text_token_id)
                    text_segment_ids.append(text_segment_id)
                    labels.append(y)
                    # 如果数据量达到批次大小或最后一个批次就进行填充并迭代出去
                    if len(text_token_ids) == self.batch_size or i == idxs[-1]:
                        # 将批量序列填充至本批次最长序列长度
                        text_token_ids = seq_padding(text_token_ids)
                        text_segment_ids = seq_padding(text_segment_ids)
                        trigger_start_index, trigger_end_index = np.array(trigger_start_index), np.array(trigger_end_index)
                        labels = np.array(labels)
                        yield [text_token_ids, text_segment_ids, trigger_start_index, trigger_end_index], labels
                        # 重现将数据各部分列表置空
                        text_token_ids, text_segment_ids = [], []
                        trigger_start_index, trigger_end_index = [], []
                        labels = []
