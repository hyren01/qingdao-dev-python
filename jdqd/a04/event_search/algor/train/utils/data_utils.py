#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月09
"""
提供模型训练过程中需要的数据加载方法和数据生成类
提供将训练数据保存成向量文件的向量生成和保存、读取方法
"""
import os
import json
import numpy as np
from tqdm import tqdm
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path


def load_data(file_path: str):
    """
    传入文件路径，按行读取文件内容，去除换行符，返回数据列表
    :param file_path: (str)数据保存路径
    :return: data(list)数据列表
    :raise:ValueError
    """
    if not os.path.exists(file_path) or os.path.isdir(file_path):
        logger.error(f"{file_path} 文件路径错误！")
        raise ValueError
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
    except TypeError:
        with open(file_path, "r", encoding="gbk") as f:
            data = f.readlines()
    # 按行读取样本
    data = [once.replace("\n", "") for once in data if once]

    return data


def generate_vec(mode: str, data: list, tokenizer, bert_model, vector_data_dir: str, vector_id_dict_dir: str):
    """
    传入保存模式、数据、分词器、模型对象、向量保存文件夹路径、向量保存索引文件夹
    :param mode: （str）数据模式 train dev test
    :param data: (list) 数据列表 sentence sentence2 label
    :param tokenizer: 分字器
    :param bert_model: bert模型对象
    :param vector_data_dir: （str）向量保存文件夹
    :param vector_id_dict_dir: (str)事件{cameo:[id]}保存文件夹
    :return: None
    """
    # 下标列表
    idxs = list(range(len(data)))
    # 数据字典
    data_dict = {}
    # 保存向量的列表
    X = []

    for j, i in tqdm(enumerate(idxs)):
        d = data[i]
        # 匹配样本的两个句子
        text_01 = d.split("	")[0]
        text_02 = d.split("	")[1]
        # 按照句子对的数量保存索引，每10000个句子对保存成一个文件
        # 每个向量文件对应一个索引列表，以j//10000为键，以句子列表为值
        data_dict.setdefault(j // 10000, [])
        # 如果短句不在字典中，则将短句保存到字典中，并将向量化的短句保存到向量文件中
        if text_01 not in data_dict[j // 10000]:
            # 将短句保存到字典中
            data_dict[j // 10000].append(text_01)
            # 将短句向量化并保存到向量列表中
            x1_1, x1_2 = tokenizer.encode(first_text=text_01)
            x1 = bert_model.model.predict([np.array([x1_1]), np.array([x1_2])])[0][0]

            X.append(x1)
        # 同上
        if text_02 not in data_dict[j // 10000]:
            data_dict[j // 10000].append(text_02)
            x2_1, x2_2 = tokenizer.encode(first_text=text_02)
            x2 = bert_model.model.predict([np.array([x2_1]), np.array([x2_2])])[0][0]

            X.append(x2)

        # 如果达到10000或者最后一个批次则将向量保存，置空变量再次循环
        if (j + 1) % 10000 == 0 or (j + 1) == len(data):
            X = np.array(X)
            file_name = "{}_{}.npy".format(mode, j // 10000)
            file_path = cat_path(vector_data_dir, file_name)
            np.save(file_path, X)
            X = []

    # 最后将字典文件保存
    dict_name = "{}_dict.json".format(mode)
    dict_path = cat_path(vector_id_dict_dir, dict_name)

    with open(dict_path, "w", encoding="utf-8") as f:
        content = json.dumps(data_dict, ensure_ascii=False, indent=4)
        f.write(content)


def load_vec_data(vector_data_dir: str):
    """
    传入向量数据保存的文件夹，加载所有的向量数据到内存中。
    :param vector_data_dir: (str)向量数据保存的文件夹
    :return: vec_data(dict)向量数据字典
    :raise:ValueError 保存向量的文件夹路径不是文件夹或者不存在
    """
    if not os.path.exists(vector_data_dir) or os.path.isfile(vector_data_dir):
        logger.error(f"{vector_data_dir}向量保存文件夹错误！")
        raise ValueError
    # 将所有向量加载到内存中
    vec_data = {}
    file_list = os.listdir(vector_data_dir)
    for file in file_list:
        vec_data["{}".format(file)] = np.load(cat_path(vector_data_dir, file))

    return vec_data


class DataGenerator(object):
    """
    构建数据生成类，传入数据，实现向量数据迭代输出。
    1、根据传入的模式名称，得到保存数据索引字典的文件名称{mode}_dict.json
    2、根据传入的字典文件夹拼接字典字典名称，获取字典保存路径
    3、根据data中句子在字典中保存的列表的下标到vec_data向量数据中获取该句子的向量
    4、根据将得到的向量数据保存到列表中，每个批次以迭代的形式输出，将该类构建成数据生成器
    """

    def __init__(self, mode, data, vec_data, vector_id_dict_dir, batch_size, shuffle=True):
        """
        构造方法，传入类的实例变量
        :param mode: (str) 数据模式 train dev test
        :param data: (list) 传入的数据列表
        :param vec_data: (dict) 所有的{id:[vec]}
        :param vector_id_dict_dir: (str) 向量id保存字典文件夹
        :param batch_size: (int)批量大小
        :param shuffle: (bool)是否打乱
        """
        self.mode = mode
        self.data = data
        self.vec_data = vec_data
        self.vector_id_dict_dir = vector_id_dict_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 数据的步数
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        # 保存数据对应id字典的文件名称
        dict_name = f"{self.mode}_dict.json"
        # 字典保存路径
        dict_path = cat_path(vector_id_dict_dir, dict_name)
        # 获取字典
        with open(dict_path, "r", encoding="utf-8") as f:
            self.data_dict = f.read()
            self.data_dict = json.loads(self.data_dict)

    def __len__(self):
        """
        :return:返回数据集步数
        """
        return self.steps

    def __iter__(self):
        """
        构造迭代方法
        """
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            vectors_1, vectors_2, labels = [], [], []
            for i in idxs:
                # 获取需要进行相似度匹配的两个短句以及标签
                x1 = self.data[i].split("\t")[0]
                x2 = self.data[i].split("\t")[1]
                y = int(self.data[i].split("\t")[2])
                # 根据保存是的规则，每10000个句子对保存一个列表，以i//10000值为键，获取保存句子的列表
                key = str(i // 10000)
                # 到列表中获取句子的索引，用于到向量列表中获取对应的向量
                x1_id = self.data_dict[key].index(x1)
                x2_id = self.data_dict[key].index(x2)

                # 训练集向量保存位置
                file_name = f"{self.mode}_{key}.npy"
                # 根据句子的索引获取句子在向量文件中的向量
                x1 = self.vec_data[file_name][x1_id]
                x2 = self.vec_data[file_name][x2_id]

                vectors_1.append(x1)
                vectors_2.append(x2)
                labels.append(y)
                # 如果达到批量大小或者是最后一个批次则迭代出去
                if len(vectors_1) == self.batch_size or i == idxs[-1]:
                    labels = np.array(labels)
                    vectors_1 = np.array(vectors_1)
                    vectors_2 = np.array(vectors_2)
                    yield [vectors_1, vectors_2], labels
                    vectors_1, vectors_2, labels = [], [], []
