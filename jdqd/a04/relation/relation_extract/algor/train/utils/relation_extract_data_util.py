# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:14:40 2020

@author: 12894
"""

import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from jdqd.common.relation_com.model_utils import seq_padding

def get_data(total_data_path, relation):
    '''
    传入数据路径，输出符合模型输入格式的训练集与测试集数据
    :param total_data_path: 全部数据路径
    return train_line(list), test_line(list)
    '''
    causality = []
    with open(total_data_path + '/' + f'{relation}_pos.txt', encoding='utf-8') as f:
        fs = list(set(f.readlines()))
        for line in fs:
            left_len = len(line.strip().split('\t')[4].split('丨'))
            right_len = len(line.strip().split('\t')[5].split('丨'))
            if left_len == right_len == 1:
                causality.append((line.strip().split('\t')[4].replace('|',''), line.strip().split('\t')[5].replace('|',''), to_categorical(1, 3)))
                causality.append((line.strip().split('\t')[5].replace('|',''), line.strip().split('\t')[4].replace('|',''), to_categorical(2, 3)))

    causality_neg = []
    with open(total_data_path + '/' + f'{relation}_neg_class.txt', encoding='utf-8') as f:
        fs = list(set(f.readlines()))
        for line in fs:
            lens = len(line.strip().split('\t')[1].split('丨'))
            if lens >= 2:
                causality_neg.append((line.strip().split('\t')[1].split('丨')[0].replace('|',''), line.strip().split('\t')[1].split('丨')[1].replace('|',''), to_categorical(0, 3)))

    causas_train = causality + random.sample(causality_neg, len(causality))
    
    random_order = list(range(len(causas_train)))
    np.random.shuffle(random_order)
    train_line = [causas_train[j] for i, j in enumerate(random_order) if i % 5 != 0]
    test_line = [causas_train[j] for i, j in enumerate(random_order) if i % 5 == 0]
    return train_line, test_line

class data_generator:
    """
    构建数据生成器，对传入的数据进行编码、shuffle、分批，迭代返回
    """
    def __init__(self, tokenizer, maxlen, data, batch_size=10, shuffle=True):
        """
        接收分字器、最大长度、数据、批量大小
        :param tokenizer: (object)分字器
        :param maxlen: (int)最大长度
        :param data: (list)数据
        :param batch_size: (int)批量大小
        :shuffle: 是否乱序
        """
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
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
            if self.shuffle:
                np.random.shuffle(idxs)#对传入的顺序进行打乱
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:self.maxlen]
                text2 = d[1][:self.maxlen]
                x1, x2 = self.tokenizer.encode(first=text1, second=text2)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], [] 
