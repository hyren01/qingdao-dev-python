# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:28:10 2020

@author: 12894
"""
import numpy as np

def get_label(label_path):
    '''
    加载数据标注字典
    :param label_path: 数据标注路径
    
    return label(dict), _label(dict)
    '''
    label = {}
    _label = {}
    f_label = open(label_path, 'r+', encoding='utf-8')
    for line in f_label:
        content = line.strip().split()
        label[content[0].strip()] = content[1].strip()
        _label[content[1].strip()] = content[0].strip()
    return label, _label


def PreProcessData(path):
    """
    标注语料格式转换
    :param path:标注数据路径
    
    return 转换结果为（sentences, tags）二元组形式
    """
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as data_file:
        for sentence in data_file.read().strip().split('\n\n'):
            _sentence = ""
            tag = []
            for word in sentence.strip().split('\n'):
                content = word.strip().split()
                if len(content) == 2:
                    _sentence += content[0]
                    tag.append(content[1])
            sentences.append(_sentence)
            tags.append(tag)
    data = (sentences, tags)
    return data


#训练集、测试集标注语料路径
def get_data(train_path, test_path, relation):
    """
    数据预处理，得到入模型的标准数据格式
    :param train_path: 训练集语料路径
    :param test_path: 测试集语料路径
    return input_train:训练集输入句子, result_train:训练集输入句子序列标签 input_test:测试集输入句子, result_test:测试集输入句子序列标签
    """
    input_train, result_train = PreProcessData(train_path + '/' + f'{relation}.train_line.txt')
    input_test, result_test = PreProcessData(test_path+ '/' + f'{relation}.test _line.txt')
    return input_train, result_train, input_test, result_test


def Vector2Id(tags):
    """
    输出softmax向量，选择最佳的Id序列标签
    :param tags: 模型输出的softmax向量
    
    return  Id序列标签
    """
    result = []
    for tag in tags:
        result.append(np.argmax(tag))
    return result

def Id2Label(ids, _label):
    """
    Id标签序列转换Label标签序列
    :param ids: Id标签序列
    
    return Label标签序列
    """
    result = []
    for id in ids:
        result.append(_label[str(id)])
    return result

