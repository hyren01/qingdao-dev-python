# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:14:40 2020

@author: 12894
"""
import os
import json
import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from jdqd.common.relation_com.model_utils import seq_padding

def sens_tagging1(sentence, index_first, index_second):
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

def sens_tagging2(sentence, index_first, index_second):
        '''
    关系事件对谓语正向标注
    :param sentence: 字符串句子
    :param index_first: 正向关系第一个谓语下标(list)
    :param index_second: 正向关系第二个谓语下标(list)
    return 标注后的句子
    '''
    if index_first[1] <= index_second[0]:
        sentence = sentence[:index_first[0]] + '#' + sentence[index_first[0]:index_first[1]] + '#' + \
            sentence[index_first[1]:index_second[0]] + '$' + sentence[index_second[0]:index_second[1]] + '$' + \
            sentence[index_second[1]:]
    else:
        sentence = sentence[:index_second[0]] + '$' + sentence[index_second[0]:index_second[1]] + '$' +\
        sentence[index_second[1]:index_first[0]] + '#' + sentence[index_first[0]:index_first[1]] + '#' + \
        sentence[index_first[1]:]
    return  sentence

def get_data_x(total_data_path, is_bi_directional = True):
    '''
    传入数据路径，输出符合模型输入格式的训练集与测试集数据
    :param total_data_path: 全部数据路径
    :is_bi_directional： 是否双向关系
    return train_line(list), test_line(list)
    '''
    relation_train = []
    dirs = os.listdir(total_data_path)
    for file in dirs:
        if file.endswith(".txt"):
            fs = open(total_data_path + '/' + file, encoding='utf-8')
            relation_train = relation_train + list(set(fs.readlines()))

    random_order = list(range(len(relation_train)))
    np.random.shuffle(random_order)
    train_data = [relation_train[j] for i, j in enumerate(random_order) if i % 5 != 0]
    test_data = [relation_train[j] for i, j in enumerate(random_order) if i % 5 == 0]
    train_line = []
    test_line = []
    if is_bi_directional == 'T':
        for line in train_data:
            if line.strip().split('\t')[-1] == '1':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((front, to_categorical(1, 3)))
                back = sens_tagging2(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((back, to_categorical(2, 3)))

            if line.strip().split('\t')[-1] == '0':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((front, to_categorical(0, 3)))
                back = sens_tagging2(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((back, to_categorical(0, 3)))
                
        for line in test_data:
            if line.strip().split('\t')[-1] == '1':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((front, to_categorical(1, 3)))
                back = sens_tagging2(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((back, to_categorical(2, 3)))

            if line.strip().split('\t')[-1] == '0':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((front, to_categorical(0, 3)))
                back = sens_tagging2(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((back, to_categorical(0, 3)))
    else:
        for line in train_data:
            if line.strip().split('\t')[-1] == '1':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((front, to_categorical(1, 2)))

            if line.strip().split('\t')[-1] == '0':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                train_line.append((front, to_categorical(0, 2)))

        for line in test_data:
            if line.strip().split('\t')[-1] == '1':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((front, to_categorical(1, 2)))

            if line.strip().split('\t')[-1] == '0':
                front = sens_tagging1(line.split('\t')[0], json.loads(line.split('\t')[3]), json.loads(line.split('\t')[4]))
                test_line.append((front, to_categorical(0, 2)))

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
                text = d[0][:self.maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], [] 
