#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月13
"""
向量法事件匹配，向量删除方法
"""
import os
import numpy as np
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
import jdqd.a04.event_search.config.PredictConfig as pre_config
from jdqd.a04.event_search.algor.predict import comm


def execute_delete(event_id: str):
    """
    删除模块的主控程序，读取cameo2id,然后查看事件id是否存在字典中，并进行删除。
    :return: None
    :raise: FileNotFoundError
    """
    if not os.path.exists(pre_config.cameo2id_path) or not os.path.isfile(pre_config.cameo2id_path):
        logger.error(f"{pre_config.cameo2id_path} miss, can not exec delete!")
        raise FileNotFoundError
    # 读取保存事件{cameo:[event_id]}字典
    cameo2id = comm.read_cameo2id(pre_config.cameo2id_path)

    # 判断事件是否在向量库中，如果存在则设置为1，并读取对应的向量并将其删除
    status = 0
    logger.info("Begin to scan cameo2id dict...")
    for cameo in list(cameo2id.keys()):

        if event_id in cameo2id[cameo]:
            status += 1
            # 读取向量的文件地址
            read_file_path = cat_path(pre_config.vec_data_dir, f"{cameo}.npy")
            # 保存向量时的路径，只是比上边缺少了.npy,函数会自动补齐
            save_file_path = cat_path(pre_config.vec_data_dir, cameo)
            # 读取cameo保存的向量文件
            x = comm.read_np(read_file_path)

            # 删除向量
            temp = np.delete(x, list(cameo2id[cameo]).index(event_id), axis=0)
            # 删除列表中的事件id
            cameo2id[cameo].remove(event_id)

            # 将更新后的文件重新保存
            # cameo对应的向量没有删除完，cameo2id[cameo]也没有删完
            if temp.shape[0] and cameo2id[cameo]:
                comm.save_cameo2id(cameo2id, pre_config.cameo2id_path)
                # 将向量保存到文件中
                comm.save_np(save_file_path, temp)
                # 跳出循环
                break

            # cameo对应的向量已经置空了，且cameo对应的列表也空了
            elif not temp.shape[0] and not cameo2id[cameo]:
                # 将向量文件删除
                os.remove(read_file_path)
                # 删除cameo在字典中的键
                del cameo2id[cameo]
                # 判断字典是否为空
                if len(cameo2id):
                    # 字典不为空则重新保存
                    comm.save_cameo2id(cameo2id, pre_config.cameo2id_path)
                else:
                    # 字典空了，则删除字典文件
                    os.remove(pre_config.cameo2id_path)
                # 跳出循环
                break

            # 说明向量表和字典不匹配，一个已经清空，而另一个没有清空，此时需要手动将对应的cameo删除
            else:
                logger.error("vector file does not match cameo2id, please delete file manually!")
                raise ValueError

    return status
