#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
存放整个项目中事件抽取、事件匹配、事件归并的文件校验、文件夹校验、数据清洗、分句等数据相关的公用方法
"""
import os
import re
import json
from feedwork.utils import logger


def valid_file(file_path: str):
    """
    传入文件路径，判断文件是否存在，存在则返回1, 不存在则返回0
    :param file_path: (str)文件路径
    :return: 1 0
    """
    # 文件状态
    state = 1
    # 判断文件是否存在
    if not os.path.exists(file_path) or os.path.isdir(file_path):
        state -= 1

    return state


def valid_dir(dir_path: str):
    """
    传入文件夹路径，判断文件夹是否存在，存在则返回1，不存在则返回0
    :param dir_path: (str)
    :return: 1,0
    """
    # 文件夹状态
    state = 1
    # 判断文件夹是否存在
    if not os.path.exists(dir_path) or os.path.isfile(dir_path):
        state -= 1

    return state


def read_json(file_path: str):
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

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except TypeError:
        with open(file_path, "r", encoding="gbk") as file:
            content = file.read()

    data = json.loads(content)

    return data


def save_json(content, file_path: str):
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


def file_reader(file_path: str):
    """
    传入文件路径，读取文件内容，以字符串方式返回文件内容。
    :param file_path: (str)文件路径
    :return: (str) content 文件内容
    :raise:
    """
    try:
        with open(file_path, 'r', encoding="gbk") as file:
            content = file.read()
    except Exception as e:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

    return content


def file_saver(content: str, file_path: str):
    """
    传入字符串内容，将内容保存到指定路径中。
    :param content: (str)文件内容
    :param file_path: (str)指定文件路径
    :return: None
    """
    assert isinstance(content, str)

    with open(file_path, "w", encoding="utf-8") as file:

        file.write(content)


def data_process(content):
    """
    对传入的中文字符串进行清洗，去除邮箱、URL等无用信息。
    :param content: (str)待清洗的中文字符串
    :return: (str) content 清洗后的中文字符串
    """
    assert isinstance(content, str)

    if content:
        content = re.sub('<.*?>', '', content)
        content = re.sub('【.*?】', '', content)
        # 剔除邮箱
        content = re.sub('([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})', '', content)
        content = re.sub('[a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+', '', content)
        # 剔除URL
        content = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",'',content)
        # 剔除16进制值
        content = re.sub('#?([a-f0-9]{6}|[a-f0-9]{3})', '', content)
        # 剔除IP地址
        content = re.sub('((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)', '', content)
        content = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '',
            content)
        # 剔除用户名密码名
        content = re.sub('[a-z0-9_-]{3,16}', '', content)
        content = re.sub('[a-z0-9_-]{6,18}', '', content)
        # 剔除网络字符，剔除空白符
        content = content.strip().strip('\r\n\t').replace(u'\u3000', '').replace(u'\xa0', '')
        content = content.replace('\t', '').replace(' ', '').replace('\n', '').replace('\r', '')

    return content


def get_sentences(content: str):
    """
    传入一篇中文文章，获取文章中的每一个句子，返回句子列表。对中文、日文文本进行拆分。
    # todo 可以考虑说话部分的分句， 例如‘xxx：“xxx。”xx，xxxx。’
    :param content: (str) 一篇文章
    :return: sentences(list) 分句后的列表
    :raise: TypeError
    """
    if not isinstance(content, str):
        logger.error("The content you want to be split is not string!")
        raise TypeError
    # 需要保证字符串内本身没有这个分隔符
    split_sign = '%%%%'
    # 替换的符号用: $PACK$
    sign = '$PACK$'
    # 替换后的检索模板
    search_pattern = re.compile('\$PACK\$')
    # 需要进行替换的模板
    pack_pattern = re.compile('(“.+?”|（.+?）|《.+?》|〈.+?〉|[.+?]|【.+?】|‘.+?’|「.+?」|『.+?』|".+?"|\'.+?\')')
    # 正则匹配文本中所有需要替换的模板
    pack_queue = re.findall(pack_pattern, content)
    # 将文本中所有需要替换的，都替换成sign替换符号
    content = re.sub(pack_pattern, sign, content)

    # 分句模板
    pattern = re.compile('(?<=[。？！])(?![。？！])')
    result = []
    while content != '':
        # 查询文章中是否可分句
        s = re.search(pattern, content)
        # 如果不可分，则content是一个完整的句子
        if s is None:
            result.append(content)
            break
        # 获取需要分句的位置
        loc = s.span()[0]
        # 将第一个句子添加到结果中
        result.append(content[:loc])
        # 将剩余的部分继续分句
        content = content[loc:]

    # 使用切分符将之前分割好的内容拼接起来
    result_string = split_sign.join(result)
    while pack_queue:
        pack = pack_queue.pop(0)
        loc = re.search(search_pattern, result_string).span()
        result_string = f"{result_string[:loc[0]]}{pack}{result_string[loc[1]:]}"

    # 使用切分符将文章内容切分成句子
    sentences = result_string.split(split_sign)

    return sentences


if __name__ == "__main__":

    content = ""

    sentences = get_sentences(content)