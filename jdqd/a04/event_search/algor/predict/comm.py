#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年07月01
# 用于向量cameo2id字典的保存和读取的方法实现，专用方法，考虑到多线程操作相同文件
import os
import time
import json
import numpy as np
from feedwork.utils import logger


def read_cameo2id(file_path: str):
    """
    传入json文件路径，读取文件内容，返回json解析后的数据。
    :param file_path: (str)json文件路径
    :return: data 读取得到的文章内容
    :raise: ValueError 如果不是json文件则报错
    """
    # 判断文件是否为.json文件
    if not os.path.exists(file_path) or os.path.isdir(file_path) or not file_path.endswith(".json"):
        logger.error(f"{file_path} 文件错误！")
        raise ValueError

    # 因为多线程，所有循环读取字典内容
    while True:

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except TypeError:
            with open(file_path, "r", encoding="gbk") as file:
                content = file.read()

        # 读取到内容则跳出循环
        if content:
            break

    data = json.loads(content)

    return data


def save_cameo2id(content, file_path: str):
    """
    传入需要保存的数据，将数据转换为json字符串保存到指定路径file_path
    :param content: 需要保存的数据
    :param file_path: (str)指定路径
    :return: None 无返回值
    :raise: ValueError 如果不是json文件则报错
    """
    # 判断是否保存为.json
    if not file_path.endswith(".json"):
        logger.error(f"{file_path} 需保存为.json文件")
        raise ValueError

    # 将列表或者字典处理成json字符串
    content = json.dumps(content, ensure_ascii=False, indent=4)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    # 休眠一秒钟，为读取线程制造时间差
    time.sleep(1)


def save_np(save_file_path, temp):
    """
    传入向量和文件名，将向量保存到文件中
    :param temp: 向量
    :param save_file_path: 保存向量的文件
    :return: None
    """
    # 保存向量文件
    np.save(save_file_path, temp)
    # 休眠一秒钟，制造时间差
    time.sleep(1)


def read_np(read_file_path):
    """
    传入文件路径，读取文件中的向量
    :param read_file_path: 保存向量的文件路径
    :return: x文件中的向量
    """
    while True:
        try:
            x = np.load(read_file_path)
            break
        except:
            continue

    return x