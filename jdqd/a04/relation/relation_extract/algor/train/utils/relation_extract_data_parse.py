#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Fan Wenhan
@Time: 2020/8/11 16:04
desc: 关系抽取的数据解析程序
"""
import os
from tqdm import tqdm
from feedwork.utils import logger
from jdqd.common.event_emm.data_utils import file_reader, valid_dir, valid_file
import itertools


def get_rl_info(file_name):
    """
    读取ann文件和txt文件，获取触发词信息
    :param file_name: 文件名（不包含后缀）
    :return: sentence: 原句子，str
    :return: rl_infos: 全部事件之间的关系信息，list
    """
    num_pos = 0
    num_neg = 0
    # all_rl 用来筛选事件之间的关系
    all_rl_type = ['Causality', 'Assumption', 'Contrast', 'Further', 'Parallel']
    # 给定标注文件
    ann_file = f"{file_name}.ann"
    txt_file = f"{file_name}.txt"
    # 获取两个标注文件的内容，其中ann_content是分行的str
    ann_content = file_reader(ann_file)
    sentence = file_reader(txt_file).strip('\n')
    lines = ann_content.split('\n')
    # all_event_rls_1 用来保存构成关系的事件对，只保存正向关系，{(T1, T2), (), ...}
    all_event_rls_1 = []
    # all_event_rls_1 用来保存构成关系的事件对，只保存逆向关系，{(T1, T2), (), ...}
    all_event_rls_2 = []
    # event_info用来保存事件谓语的索引信息，{'T1': (对应的文字, x, y), ...}
    event_info = {}
    # event_rl_infos用来保存提取出的事件关系信息
    event_rl_infos = []
    # 遍历标注文件，获取事件关系的信息，得到event_info和all_event_rls
    for line in lines:
        if line[0] == 'T':
            conts = line.split('\t')
            info = conts[1].split(' ')
            info_type = info[0]
            # 根据类型提取事件信息（谓语）
            if info_type == 'Event':
                alias = conts[0]
                ind_x = info[1]
                ind_y = info[2]
                event = conts[2].strip('\n')
                event_info[alias] = (event, ind_x, ind_y)
        if line[0] == 'R':
            conts = line.split('\t')
            info = conts[1].split(' ')
            info_type = info[0]
            if info_type in all_rl_type:
                # left表示左事件谓语，这里的left是别名，类似T1、T2...
                left = info[1][5:]
                right = info[2][5:]
                all_event_rls_1.append((left, right))
                all_event_rls_2.append((right, left))
    all_event_combs = list(itertools.permutations(list(event_info.keys()), 2))
    # 遍历全部的事件组合
    for comb in all_event_combs:
        if comb in all_event_rls_1:
            # 提取别名
            left, right = comb
            # 获取事件谓语对应的文字，索引信息
            left_cont, left_x, left_y = event_info[left]
            right_cont, right_x, right_y = event_info[right]
            # 索引信息拼成指定格式 '[x, y]'
            left_ind = f'[{left_x}, {left_y}]'
            right_ind = f'[{right_x}, {right_y}]'
            rl_info = [sentence, left_cont, right_cont, left_ind, right_ind, '1']
            rl_info = '\t'.join(rl_info) + '\n'
            event_rl_infos.append(rl_info)
            num_pos += 1
        # 反向关系的事件对跳过，算法的数据预处理会进行翻转
        elif comb in all_event_rls_2:
            continue
        else:
            # 提取别名
            left, right = comb
            # 获取事件谓语对应的文字，索引信息
            left_cont, left_x, left_y = event_info[left]
            right_cont, right_x, right_y = event_info[right]
            # 索引信息拼成指定格式 '[x, y]'
            left_ind = f'[{left_x}, {left_y}]'
            right_ind = f'[{right_x}, {right_y}]'
            rl_info = [sentence, left_cont, right_cont, left_ind, right_ind, '0']
            rl_info = '\t'.join(rl_info) + '\n'
            event_rl_infos.append(rl_info)
            num_neg += 1
    return event_rl_infos, num_pos, num_neg


def data_parse(source_dir, target_dir, verbose=False):
    """
    传入原始标注数据文件夹路径和解析后文件存放的路径，将解析好的数据保存到目标文件夹
    :param source_dir: 存放原始标注数据的文件夹
    :param target_dir: 存放解析后数据的文件夹
    :param verbose: 是否显示进度
    :return: status--解析状态， results--数据量
    """
    # 正反样本总数
    all_pos_num = 0
    all_neg_num = 0
    # 判断目标文件夹路径是否存在，不存在则创建
    if not valid_dir(target_dir):
        os.makedirs(target_dir)
    file_names = os.listdir(source_dir)
    file_names = list(set(file_name.split(".")[0] for file_name in file_names))
    with open(target_dir, 'w', encoding='utf-8') as f:
        # 显示进度
        if verbose:
            file_names = tqdm(file_names)
        # 同步遍历所有ann和txt文件，提取其中触发词信息
        for file_name in file_names:
            file_path = os.path.join(source_dir, file_name)
            # 判断两个文件是否都同时存在
            if valid_file(f"{file_path}.ann") and valid_file(f"{file_path}.txt"):
                # 解析文件获取事件关系信息
                try:
                    event_rl_infos, num_pos, num_neg = get_rl_info(file_path)
                except Exception as e:
                    logger.error(f"事件关系信息获取失败：{file_path}：{e}")
                    continue
                # 未获得事件关系信息时，跳过这次循环
                if not event_rl_infos:
                    continue
                # 遍历句子和标签序列，写入文件，文件中每一行为句子中的一个字和其对应的标签，例：但\tB-OS\n
                for event_rl_info in event_rl_infos:
                    f.write(event_rl_info)
                all_pos_num += num_pos
                all_neg_num += num_neg
            else:
                logger.error(f"ann或txt文件缺失：{file_path}")
                continue
    # 确保正负样本个数大于1
    if all_pos_num * all_neg_num != 0:
        return {"status": "success", "results": {"pos_num": all_pos_num, "neg_num": all_neg_num}}
    # 数据解析0条
    else:
        return {"status": "failed", "results": {"pos_num": all_pos_num, "neg_num": all_neg_num}}
