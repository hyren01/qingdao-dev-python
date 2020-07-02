#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
根据传入的事件cameo\event_id将向量保存到对应的向量文件中，并建立依赖于cameo和event_id索引文件
"""
import os
import numpy as np
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
import jdqd.a04.event_search.config.PredictConfig as pre_config
from jdqd.a04.event_search.algor.predict import comm


def save_vec_data(cameo: str, event_id: str, main_vec):
    """
    传入事件cameo、事件id、事件短句向量，将向量保存到文件中，临时保存，后期改写为保存到数据库中,
    将事件id存放到以cameo为键的字典中{cameo:[event_id], }
    :param cameo: (str)事件cameo号
    :param event_id: (str)事件编号
    :param main_vec: (ndarray)事件向量
    :return: None
    :raise:字典文件缺失/ 向量文件文件缺失 FileNotFoundError 事件id重复 ValueError 传入类型错误 TypeError
    """
    if not isinstance(cameo, str):
        logger.error("cameo编号格式错误!")
        raise TypeError
    if not isinstance(event_id, str):
        logger.error("事件编号格式错误!")
        raise TypeError

    # 读取向量时的路径
    read_file_path = cat_path(pre_config.vec_data_dir, f"{cameo}.npy")
    # 保存向量时的路径，只是比上边缺少了.npy,函数会自动补齐
    save_file_path = cat_path(pre_config.vec_data_dir, cameo)

    # 判断字典文件是否存在, 不存在这就是第一个向量
    if not os.path.exists(pre_config.cameo2id_path):
        cameo2id = {cameo: [event_id]}
        # 将字典保存
        comm.save_cameo2id(cameo2id, pre_config.cameo2id_path)

        # 将向量保存到文件中
        comm.save_np(save_file_path, np.array([main_vec]))

    # 如果字典文件存在，而向量文件不存在，则说明这是这个cameo的第一个向量
    elif not os.path.exists(read_file_path):
        # cameo2id 字典 {cameo:[]}
        cameo2id = comm.read_cameo2id(pre_config.cameo2id_path)
        # 将事件id添加到字典中
        cameo2id[cameo] = [event_id]
        # 将向量保存到文件中
        comm.save_np(save_file_path, np.array([main_vec]))

        # 写入文件中
        comm.save_cameo2id(cameo2id, pre_config.cameo2id_path)

    # 如果向量文件存在，而字典文件不存在，则报错，字典文件缺失
    elif not os.path.exists(pre_config.cameo2id_path):
        logger.error("字典文件缺失!")
        raise FileNotFoundError

    # 如果字典和向量都存在，则正常添加向量和事件id，并保存
    else:
        # cameo2id 字典 {cameo:[]}
        cameo2id = comm.read_cameo2id(pre_config.cameo2id_path)

        # 如果事件id不在字典中，则进行保存向量和添加id索引的操作
        if event_id not in cameo2id.setdefault(cameo, []):
            # 将事件id保存到cameo2id字典中
            cameo2id[cameo].append(event_id)

            # 读取向量文件
            x = comm.read_np(read_file_path)
            # 将向量拼接进去
            temp = np.vstack([x, np.array([main_vec])])
            # 将向量保存到文件中
            comm.save_np(save_file_path, temp)

            # 写入文件中
            comm.save_cameo2id(cameo2id, pre_config.cameo2id_path)

        # 事件id已经存在了，说明输入的id重复了，则报错，不进行保存操作。
        else:
            logger.error("事件id重复！")
            raise ValueError
