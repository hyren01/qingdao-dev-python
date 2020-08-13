#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: Fan Wenhan
@Time: 2020/8/11 16:03
desc: 触发词抽取的数据解析程序
"""
import os
from tqdm import tqdm
from feedwork.utils import logger
from jdqd.common.event_emm.data_utils import file_reader, valid_dir, valid_file

# 序列标注规则，用于区分不同的触发词类别
type_dict = {'causality': 'C', 'assumption': 'A', 'contrast': 'O', 'further': 'F', 'parallel': 'P'}


def data_trans(sentence, kw_infos):
    """
    根据句子和触发词信息，生成标签序列
    :param sentence: 原始句子，str
    :param kw_infos: 触发词信息，元组列表
    :return: tag: 标签序列，list
    :return: kw_flag: 是否包含触发词标志，bool
    """
    sent_len = len(sentence)
    # 这个是字母o不是零
    tag = ['O'] * sent_len
    # 触发词标志，表示是否包含触发词
    kw_flag = True
    # 遍历触发词信息，修改tag
    for kw_info in kw_infos:
        # 单个关键词kw_info->(类型，x，y)
        if len(kw_info) == 3:
            kw_type = kw_info[0]
            kword_coord1 = kw_info[1]
            kword_coord2 = kw_info[2]
            tag[kword_coord1:kword_coord2] = [f'B-{type_dict[kw_type]}S'] + (kword_coord2 - kword_coord1 - 1) * [
                f'I-{type_dict[kw_type]}S']
        # 两个关键词kw_info->(类型，左词x，左词y，右词x，右词y)
        elif len(kw_info) == 5:
            kw_type, kword_coord01, kword_coord02, kword_coord11, kword_coord12 = kw_info
            tag[kword_coord01:kword_coord02] = [f'B-{type_dict[kw_type]}C'] + (kword_coord02 - kword_coord01 - 1) * \
                                               [f'I-{type_dict[kw_type]}C']
            tag[kword_coord11:kword_coord12] = [f'B-{type_dict[kw_type]}E'] + (kword_coord12 - kword_coord11 - 1) * \
                                               [f'I-{type_dict[kw_type]}E']
        else:
            # 出现错误的长度
            kw_flag = False
            break
    return tag, kw_flag


def get_kw_ind(file_name):
    """
    读取ann文件和txt文件，获取触发词信息
    :param file_name: 文件名（不包含后缀）
    :return: sentence: 原句子，str
    :return: kw_infos: 全部的触发词信息，list
    """
    # 因为文件中同时包含触发词抽取和关系抽取的标注信息，通过触发词类别筛选触发词抽取语料
    all_kw_type = ['Causality', 'Assumption', 'Contrast', 'Further', 'Parallel']
    # 给定标注文件
    ann_file = f"{file_name}.ann"
    txt_file = f"{file_name}.txt"
    # 获取两个标注文件的内容，其中ann_content是分行的str
    ann_content = file_reader(ann_file)
    sentence = file_reader(txt_file).strip('\n')
    # 触发词信息列表（已配对） [(类型，左词x，左词y，右词x，右词y), (类型，单词x，单词y), ...]
    kw_infos = []
    lines = ann_content.split('\n')
    # 触发词信息（未配对），{alias1:(type1, x1, y1), alias2:(type2, x2, y2), ...}
    kw_dict = {}
    # kw_rls用来保存匹配成对的触发词，元组列表
    kw_rls = []
    for line in lines:
        # 第一个字符为"T"表示这行可能是触发词信息，也可能是事件
        # TODO 这里默认触发词和事件（谓语）的别名都是T，因此通过后面的类型进行区分，是否可以改成不同别名，需后续验证
        if line[0] == 'T':
            conts = line.split('\t')
            alias = conts[0]
            info = conts[1].split(' ')
            info_type = info[0]
            if info_type in all_kw_type:
                ind_x = info[1]
                ind_y = info[2]
                kw_dict[alias] = (info_type, ind_x, ind_y)
        # 第一个字符为"R"表示这行是触发词之间的关系信息
        # TODO 这里还不确定事件之间的关系是不是R，暂时未进行区分
        elif line[0] == 'R':
            conts = line.split('\t')
            info = conts[1].split(' ')
            # left表示左词，例如因果关系中的因为，right表示右词，例如因果关系中的所以，这里的left是别名，类似T1、T2...
            left = info[1][5:]
            right = info[2][5:]
            kw_rls.append((left, right))
    # 将触发词进行匹配
    for kw_rl in kw_rls:
        # 此处暂时不加判断，判断关系是否是触发词的关系，利用首字母（T、R...判断）
        left, right = kw_rl
        # 构成关系的两个触发词中至少有一个词不是触发词，跳过该次遍历
        if left not in kw_dict or right not in kw_dict:
            continue
        # 两个触发词属于不同类别，这种情况说明语料标注有误
        if kw_dict[left][0] != kw_dict[right][0]:
            logger.error(f"标注错误，建立关系的两个触发词的类别不同：{file_name}")
            continue
        kw_infos.append((kw_dict[left][0], kw_dict[left][1], kw_dict[left][2], kw_dict[right][1], kw_dict[right][2]))
        # 将成对的触发词从触发词信息中删除
        kw_dict.pop(left)
        kw_dict.pop(right)
    # 添加单独的触发词
    for kw in kw_dict:
        kw_infos.append(kw_dict[kw])
    return sentence, kw_infos


def data_parse(source_dir, target_dir, verbose=False):
    """
    传入原始标注数据文件夹路径和解析后文件存放的路径，将解析好的数据保存到目标文件夹
    :param source_dir: 存放原始标注数据的文件夹
    :param target_dir: 存放解析后数据的文件夹
    :param verbose: 是否显示进度
    :return: status--解析状态， results--数据量
    """
    # 语料中的句子数量
    all_sentence_num = 0
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
                # 解析文件获取关系触发词的索引
                try:
                    sentence, kw_infos = get_kw_ind(file_path)
                except Exception as e:
                    logger.error(f"触发词信息获取失败：{file_path}：{e}")
                    continue
                # 未获得触发词信息时，跳过这次循环
                if not kw_infos:
                    continue
                # 根据触发词位置信息，生成标签序列tag->['O', 'O', 'B-CS', 'I-CS', ...]
                try:
                    tag, kw_flag = data_trans(sentence, kw_infos)
                except Exception as e:
                    logger.error(f"生成标签序列失败：{file_path}：{e}")
                    continue
                # 不包含触发词时，跳出循环，这种情况一般是kw_infos中的触发词信息元组长度有问题，长度只能是3和5
                if not kw_flag:
                    continue
                # 遍历句子和标签序列，写入文件，文件中每一行为句子中的一个字和其对应的标签，例：但\tB-OS\n
                for i in range(len(sentence)):
                    f.write(sentence[i] + '\t' + tag[i] + '\n')
                f.write('\n')
                all_sentence_num += 1
            else:
                logger.error(f"ann或txt文件缺失：{file_path}")
                continue
    if all_sentence_num > 0:
        return {"status": "success", "results": {"sentences": all_sentence_num}}
    # 数据解析0条
    else:
        return {"status": "failed", "results": {"sentences": all_sentence_num}}
