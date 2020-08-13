# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:28:10 2020

@author: 12894
"""
import numpy as np


"""
全部的关键词标签
标签说明，B表示关键词的begin，I表示inside,"-"后面的部分表示标签类别，由两个字母组成，第一个字母表示关键词类别，
第一个字母说明：C-因果，A-假设，O-转折，F-递进，P-并列
第二个字母说明：S-单独的关键词，C-成对的关键词的左词（例如因果中的“因为”），E-成对的关键词中的右词（例如因果中的“所以”）
"""
all_tags = [("B-CS", "I-CS"), ("B-CC", "I-CC"), ("B-CE", "I-CE"), ("B-AS", "I-AS"), ("B-AC", "I-AC"), ("B-AE", "I-AE"),
        ("B-OS", "I-OS"), ("B-OC", "I-OC"), ("B-OE", "I-OE"), ("B-FS", "I-FS"), ("B-FC", "I-FC"), ("B-FE", "I-FE"),
        ("B-PS", "I-PS"), ("B-PC", "I-PC"), ("B-PE", "I-PE")]


def get_label(label_path):
    """
    加载数据标注字典
    :param label_path: 数据标注路径
    return label(dict), _label(dict)
    """
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


# 训练集、测试集标注语料路径
def get_data(train_path, test_path, relation):
    """
    数据预处理，得到入模型的标准数据格式
    :param train_path: 训练集语料路径
    :param test_path: 测试集语料路径
    return input_train:训练集输入句子, result_train:训练集输入句子序列标签 input_test:测试集输入句子, result_test:测试集输入句子序列标签
    """
    input_train, result_train = PreProcessData(train_path + '/' + f'ner_{relation}_train_line.txt')
    input_test, result_test = PreProcessData(test_path + '/' + f'ner_{relation}_test_line.txt')
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


def _find_tag(labels, B_label, I_label):
    """
    查找关键词索引
    :param labels: 标签列表，['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :param B_label: 关键词的开始标签
    :param I_label: 关键词标签
    :return: 关键词索引信息，list，[(start_pos1, len1), (start_pos2, len2), ...]
    """
    result = []
    if isinstance(labels, str):
        labels = labels.strip().split()
        labels = ["O" if label == "0" else label for label in labels]
    for num in range(len(labels)):
        if labels[num] == B_label:
            # pos0为关键词的其实索引
            pos0 = num
        if labels[num] == I_label and labels[num - 1] == B_label:
            # lenth是关键词的长度
            lenth = 2
            for num2 in range(num, len(labels)):
                if labels[num2] == I_label and labels[num2 - 1] == I_label:
                    lenth += 1
                # 标签为“O”或超过总长度时，停止遍历
                if labels[num2] == "O" or num2 == len(labels) - 1:
                    result.append((pos0, lenth))
                    break
        if labels[num] == 'O' and labels[num - 1] == B_label:
            # 这种情况针对单个字的关键词，例如“但”，对应的标签就只有“B-CS”，没有“I-CS”
            result.append((pos0, 1))
            break
    return result


def find_all_tag(labels):
    """
    查找所有的关键词索引
    :param labels: 标签列表，['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :return: 全部关键词的索引信息，字典，{CS：[(pos, len), (pos, len)], CC:[...], ...}
    """
    result = {}
    for tag in all_tags:
        res = _find_tag(labels, B_label=tag[0], I_label=tag[1])
        result[tag[0].split("-")[1]] = res
    return result


def cal_precision(pre_labels, true_labels):
    """
    计算关键词识别的精确率，根据关键词的索引信息进行匹配计算
    :param pre_labels: 预测的序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :param true_labels: 真实序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :return: 精确率，如果预测全部为"O"，精确率为0
    """
    pre = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label == "0" else label for label in pre_labels]
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label == "0" else label for label in true_labels]
    # 提取关键词的索引信息
    pre_result = find_all_tag(pre_labels)
    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    if pre:
        return sum(pre) / len(pre)
    else:
        return 0


def cal_recall(pre_labels, true_labels):
    """
    计算关键词识别的召回率
    :param pre_labels: 预测的序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :param true_labels: 真实序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :return: 召回率，如果预测全部为"O"，召回率为1
    """
    recall = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label == "0" else label for label in pre_labels]
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label == "0" else label for label in true_labels]

    true_result = find_all_tag(true_labels)
    for name in true_result:
        for x in true_result[name]:
            if x:
                if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    recall.append(1)
                else:
                    recall.append(0)
    if recall:
        return sum(recall) / len(recall)
    else:
        return 1


def cal_f1_score(precision, recall):
    """
    计算关键词识别的f1
    :param pre_labels: 预测的序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :param true_labels: 真实序列， ['O', 'B-CS', 'I-CS', 'O', ...]，注意，这里的是字母O，不是零
    :return: f1值
    """
    if precision == 0 and recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)
