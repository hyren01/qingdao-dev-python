#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
提供事件抽取模型训练数据加载方法和数据生成类
"""
import os
import numpy as np
from random import choice
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
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回
    """

    def __init__(self, tokenizer, maxlen, data, batch_size=8):
        """
        接收分字器、最大长度、数据、批量大小
        :param tokenizer: (object)分字器
        :param maxlen: (int)最大长度
        :param data: (list)数据
        :param batch_size: (int)批量大小
        """
        self.data = data
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # 计算数据步数
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        """
        :return:返回该数据集的步数
        """
        return self.steps

    def __iter__(self):
        """
        构造生成器
        :return: 迭代返回批量数据
        """
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)  # 对传入的数据进行打乱

            # 初始化数据列表
            # 字符串编码
            text_token_ids, text_segment_ids = [], []
            # 动词标签值编码
            trigger_start_label, trigger_end_label = [], []
            # 动词下标
            trigger_start_index, trigger_end_index = [], []
            # 宾语标签值编码
            object_start_label, object_end_label = [], []
            # 主语标签值编码
            subject_start_label, subject_end_label = [], []
            # 地点标签值编码
            loc_start_label, loc_end_label = [], []
            # 时间标签值编码
            time_start_label, time_end_label = [], []
            # 否定词标签值编码
            negative_start_label, negative_end_label = [], []

            # 遍历下标打乱后的下标列表
            for i in idxs:
                d = self.data[i]
                text = d['sentence'][:self.maxlen]
                # 对句子进行分字
                tokens = self.tokenizer.tokenize(text)

                items = {}
                # 根据得到的触发词，抽取相应的论元组成部分
                for event in d['events']:
                    # 对动词进行分字
                    trigger_token = self.tokenizer.tokenize(event['trigger'][0][0])[1:-1]
                    # 从数据集中获取动词起始下标
                    triggerid = int(event['trigger'][0][1][0]) + 1
                    # 动词在分字后的句子中的起始下标和终止下标，因为动词在事件中是唯一的
                    # 使用动词下标元组作为一个事件的键，其他论元组成部分作为值
                    key = (triggerid, triggerid + len(trigger_token))

                    # 如果动词下标不在事件元素集合中，则添加进去
                    if key not in items:
                        items[key] = []
                    # 初始化主体、客体、地点、时间、否定词下标列表
                    subject_ids, object_ids, loc_ids, time_ids, privative_ids = [], [], [], [], []

                    if event['subject']:  # 主语
                        for p in event['subject']:
                            # 将事件主语下标保存到列表中
                            subject_token = self.tokenizer.tokenize(p[0])[1:-1]
                            subject_id = int(p[1][0]) + 1
                            subject_ids.append((subject_id, subject_id + len(subject_token)))

                    if event['object']:  # 宾语
                        for o in event['object']:
                            # 将事件宾语下标保存到列表中
                            object_token = self.tokenizer.tokenize(o[0])[1:-1]
                            object_id = int(o[1][0]) + 1
                            object_ids.append((object_id, object_id + len(object_token)))

                    if event['loc']:  # 地点
                        for l in event['loc']:
                            # 将事件地点下标保存到列表中
                            loc_token = self.tokenizer.tokenize(l[0])[1:-1]
                            loc_id = int(l[1][0]) + 1
                            loc_ids.append((loc_id, loc_id + len(loc_token)))

                    if event['time']:  # 时间
                        for t in event['time']:
                            # 将事件发生时间下标保存到列表中
                            time_token = self.tokenizer.tokenize(t[0])[1:-1]
                            time_id = int(t[1][0]) + 1
                            time_ids.append((time_id, time_id + len(time_token)))

                    if event['privative']:  # 否定词
                        for n in event['privative']:
                            # 将事件否定词下标保存到列表中
                            privative_token = self.tokenizer.tokenize(n[0])[1:-1]
                            privative_id = int(n[1][0]) + 1
                            privative_ids.append((privative_id, privative_id + len(privative_token)))
                    # 将所有的组成部分以触发词下标为键构成字典
                    items[key].append((subject_ids, object_ids, loc_ids, time_ids, privative_ids))

                # 如果句子中事件不为空则开始进行ids化
                if items:
                    # 将事件句子进行ids化编码[1,4,5,3,7,88,98,88,77,65,5][0,0,0,0,0,0,0,0,0]
                    t1, t2 = self.tokenizer.encode(first_text=text)
                    text_token_ids.append(t1)
                    text_segment_ids.append(t2)
                    # 创建全0数组，用于构造动词起始下标、终止下标的标签值，
                    # [0,0,0,0,0,0,0,0,][0,0,0,0,0,0,0,0,]
                    d1, d2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                    # 在动词的起点位置和终点位置赋值为1，作为动词标签值
                    for j in items:
                        # 如果因为标注问题导致下标越界，则跳过本次样本构造，进入下一个事件
                        try:
                            d1[j[0]] = 1
                            d2[j[1] - 1] = 1
                        except IndexError:
                            d1, d2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                            continue
                    # 将动词的起始下标和终止下标分别赋值给k1和k2
                    k1, k2 = np.array(list(items.keys())).T
                    # 随机抽取动词，目的是增强模型鲁棒性
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    # 同理，按照构造动词下标输出的方式，构造其他论元标签值
                    # 宾语标签值
                    o1, o2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    # 主语标签值
                    p1, p2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    # 地点标签值
                    l1, l2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    # 时间标签值
                    tm1, tm2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                    # 否定词标签值
                    n1, n2 = np.zeros((len(tokens))), np.zeros((len(tokens)))

                    # 遍历事件字典，获取事件论元的起始、终止下标，并在对应的标签值位置赋值1
                    for e in items.get((k1, k2), []):
                        # 如果在获取下标时存在因最大长度不允许或者标注错误导致的下标越界，
                        # 则返回继续构造下一事件样本
                        try:
                            for j in e[0]:  # 主语
                                # 获取主语下标，构造主语标签值
                                p1[j[0]] = 1
                                p2[j[1] - 1] = 1
                            for j in e[1]:  # 宾语
                                # 获取宾语下标，构造宾语标签值
                                o1[j[0]] = 1
                                o2[j[1] - 1] = 1
                            for j in e[2]:  # 地点
                                # 获取地点下标，构造宾语标签值
                                l1[j[0]] = 1
                                l2[j[1] - 1] = 1
                            for j in e[3]:  # 时间
                                # 获取时间下标，构造时间标签值
                                tm1[j[0]] = 1
                                tm2[j[1] - 1] = 1
                            for j in e[4]:  # 否定词
                                # 获取否定词下标，构造否定词标签值
                                n1[j[0]] = 1
                                n2[j[1] - 1] = 1
                        except IndexError:
                            o1, o2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            p1, p2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            l1, l2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            tm1, tm2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            n1, n2 = np.zeros((len(tokens))), np.zeros((len(tokens)))
                            continue

                    # 将处理好的事件论元标签存放到各自的批量列表中
                    trigger_start_label.append(d1)
                    trigger_end_label.append(d2)
                    trigger_start_index.append([k1])
                    trigger_end_index.append([k2 - 1])
                    object_start_label.append(o1)
                    object_end_label.append(o2)
                    subject_start_label.append(p1)
                    subject_end_label.append(p2)
                    loc_start_label.append(l1)
                    loc_end_label.append(l2)
                    time_start_label.append(tm1)
                    time_end_label.append(tm2)
                    negative_start_label.append(n1)
                    negative_end_label.append(n2)
                    # 如果数据量达到批次大小或最后一个批次就进行填充并迭代出去
                    if len(text_token_ids) == self.batch_size or i == idxs[-1]:
                        # 序列填充，批量数据填充至于最大序列长度等长的长度
                        text_token_ids = seq_padding(text_token_ids)  # 原始句子编码
                        text_segment_ids = seq_padding(text_segment_ids)
                        trigger_start_label = seq_padding(trigger_start_label)  # 动词标签
                        trigger_end_label = seq_padding(trigger_end_label)
                        object_start_label = seq_padding(object_start_label)  # 宾语标签
                        object_end_label = seq_padding(object_end_label)
                        subject_start_label = seq_padding(subject_start_label)  # 主语标签
                        subject_end_label = seq_padding(subject_end_label)
                        loc_start_label = seq_padding(loc_start_label)  # 地点标签
                        loc_end_label = seq_padding(loc_end_label)
                        time_start_label = seq_padding(time_start_label)  # 时间标签
                        time_end_label = seq_padding(time_end_label)
                        negative_start_label = seq_padding(negative_start_label)  # 否定词标签
                        negative_end_label = seq_padding(negative_end_label)
                        trigger_start_index, trigger_end_index = np.array(trigger_start_index), np.array(
                            trigger_end_index)  # 动词下标

                        yield [text_token_ids, text_segment_ids, trigger_start_label, trigger_end_label,
                               trigger_start_index, trigger_end_index, object_start_label, object_end_label,
                               subject_start_label, subject_end_label, loc_start_label, loc_end_label, time_start_label,
                               time_end_label, negative_start_label, negative_end_label], None
                        # 重新将数据各部分列表置空
                        text_token_ids, text_segment_ids = [], []
                        trigger_start_label, trigger_end_label = [], []
                        trigger_start_index, trigger_end_index = [], []
                        object_start_label, object_end_label = [], []
                        subject_start_label, subject_end_label = [], []
                        loc_start_label, loc_end_label = [], []
                        time_start_label, time_end_label = [], []
                        negative_start_label, negative_end_label = [], []
