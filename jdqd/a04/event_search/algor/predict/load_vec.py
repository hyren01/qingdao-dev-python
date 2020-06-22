#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
根据传入的cameo, 加载向量化后的事件向量
"""
import os
import json
import numpy as np
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
import jdqd.a04.event_search.config.PredictConfig as pre_config


def load_vec_data(cameo: str):
    """
    传入事件cameo号，到cameo号对应的列表中加载所有事件短句向量
    :param cameo:(str)事件cameo号
    :return:data(dict){事件id:向量}
    :raise:TypeError FileNotFoundError
    """
    # 需要读取向量的文件路径
    read_file_path = cat_path(pre_config.vec_data_dir, f"{cameo}.npy")

    data = {}
    # 判断文件是否存在，不存在则返回空值
    if not os.path.exists(read_file_path):
        return data
    # 如果字典文件缺失则报错
    elif not os.path.exists(pre_config.cameo2id_path):
        logger.error("cameo映射事件向量的字典文件缺失!")
        raise FileNotFoundError

    else:
        with open(pre_config.cameo2id_path, "r", encoding="utf-8") as f:
            cameo2id = f.read()
        # cameo2id 字典 {cameo:[]}
        cameo2id = json.loads(cameo2id)

        # 读取文件中的向量
        x = np.load(read_file_path)

        logger.info(f"开始加载向量数据{read_file_path}。。。")
        for key, value in zip(cameo2id[cameo], x):
            data[key] = value
        logger.info(f"{read_file_path}向量数据加载完成！")

        return data
