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


def list_find(list1, list2):
    """
    在序列list1中寻找子串list2,如果找到，返回第一个下标；
    如果找不到，返回-1
    :param list1: (list, str)
    :param list2: (list, str)
    :return: i(int)起始下标， -1 未找到
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


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

    def normalize_num_str(content: str):
        """
        传入字符串，对字符串中的数字字符串进行处理，解决 1,000 类字符串中逗号引起的模型抽取问题
        :param content: 文本字符串
        :return: content(str)数字字符串规范化以后的文本内容
        """
        # 正则模板
        p = re.compile(r'(\d[,，]{1}\d)')
        # 查找所有的模板匹配组
        m = p.findall(content)
        for once in m:
            content = content.replace(once, re.sub('[,，]', "", once))

        return content

    if content:
        # 将文章中字符串内容的数字进行规范化，去除数字中的逗号
        content = normalize_num_str(content)
        # 清洗url标签
        content = re.sub('<.*?>', '', content)
        content = re.sub('【.*?】', '', content)
        # 剔除邮箱
        content = re.sub('([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})', '', content)
        content = re.sub('[a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+', '', content)
        # 剔除URL
        content = re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?", '', content)
        # 剔除IP地址
        content = re.sub('((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?)', '', content)
        content = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
                         '',
                         content)
        # 剔除网络字符，剔除空白符
        content = content.strip().strip('\r\n\t').replace(u'\u3000', '').replace(u'\xa0', '')
        content = content.replace('\t', '').replace(' ', '').replace('\n', '').replace('\r', '')

    return content


def get_sentences(content: str, maxlen=160):
    """
    传入一篇中文文章，获取文章中的每一个句子，返回句子列表。对中文、日文文本进行拆分。
    # todo 可以考虑说话部分的分句， 例如‘xxx：“xxx。”xx，xxxx。’
    :param content: (str) 一篇文章
    :return: sentences(list) 分句后的列表
    :raise: TypeError
    """

    def supplement(sentences: list):
        """
        传入句子列表，判断句子是否因正则切分导致过长，使用暴力切分方式进行分句
        :param sentences: 句子列表
        :return: results(list)
        """
        data = []
        for once in sentences:
            if len(once) > maxlen:
                data.extend([f"{i}。" for i in re.split('[。？！?!]', once) if i])
            else:
                data.append(once)

        return data

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
    sentences = supplement(result_string.split(split_sign))

    return sentences


def supplement_nums(part: str, s: str):
    """
    传入子串和原始字符串，不全子串前后的数值
    :param part: 子串
    :param s: 原始字符串
    :return: part(str)补全后的子串
    """
    start = list_find(s, part)
    end = start + len(part)

    if s[start] != s[0]:
        if s[start - 1].isdigit():
            for i in s[start - 1::-1]:
                if not i.isdigit():
                    break
                part = f"{i}{part}"

    if s[end - 1] != s[-1]:
        if s[end].isdigit():
            for i in s[end:]:
                if not i.isdigit():
                    break
                part = f"{i}{part}"

    return part


if __name__ == "__main__":
    content = """
    俄罗斯外交部无任所大使科尔丘诺夫28日在接受卫星通讯社采访时表示，俄罗斯在北极的活动与军事和政治局势是相称的。 斯普特尼克/弗拉基米尔·阿斯塔波维奇俄罗斯原子能公司:新系列核破冰船中的最后一艘“楚科奇”号将于2021年建造。 科尔舒诺夫说:“俄罗斯在北极地区升级科尔舒诺夫武装力量和进行作战训练的活动不是多余的，而是防御性的。” 与新出现的军事政治形势相称，不对北极国家的国家安全构成威胁，不违反任何国际法协议。 其目的之一是确保北纬地区的生态安全，救援和科学工作。 7月28（俄罗斯卫星通讯社）--俄罗斯外交部无任所大使尼古拉·科尔丘诺夫也强调，“俄罗斯从未在其他北极国家领土上部署”俄罗斯自己的士兵，也没有“俄罗斯提供”俄罗斯自己的领土供其他国家部署“军队”。 此外，俄罗斯也没有在北极地区与非北极国家进行军事演习，因为非北极国家在高纬度地区的军事活动只会削弱地区安全，加剧冲突和紧张，“7月28号（俄罗斯卫星通讯社）--俄罗斯外交部无任所大使尼古拉·科尔丘诺夫指出”。 早些时候，美国负责欧洲和欧亚事务的第一副国务卿迈克尔·墨菲说，俄罗斯在北极增加军事存在“超出了防御范围”，五角大楼和俄罗斯的盟友应该做出回应。特别是7月28号（俄罗斯卫星通讯社），俄罗斯外交部无任所大使尼古拉·科尔丘诺夫提到了俄罗斯将建立新的北极司令部和北极支队，重建港口和机场，建设新的基础设施，并计划在科拉半岛部署S-400系统。 科尔丘诺夫说，美国没有就租借或出售破冰船问题与俄罗斯进行过接触。 此外，在建造自己舰队的同时，指示美国部门研究从伙伴国包租破冰船的可能性，Korchunov Korchunov说。 美国专家认为，俄罗斯首先包括芬兰，瑞典和加拿大。 美国人没有就租赁或出售破冰船一事与我们联系。 “美国总统特朗普下令计划为北极和南极洲建造破冰船”。 为了“支持这些地区的国家利益”，计划发展一支破冰船舰队，其中将包括至少三艘重型船和几艘中型破冰船。 此外，科尔丘诺夫表示，在俄北极岛屿部署S-400防空导弹是自然的一步，完全是出于防御的需要。 至于S-400防空导弹在俄罗斯北极岛屿上的存在，这是继在我国领土上建立密集雷达网之后的自然步骤，“科尔丘诺夫说。”。 S-400防空导弹系统是专为防御而设的，而且无论S-400防空导弹系统部署在哪里，S-400防空导弹系统都不应该引起不安，当然，如果表示这种不安的人不是对S-400防空导弹所保护的区域或设施别有用心的话。
    
    """
    sentences = get_sentences(data_process(content))

    print(sentences)

    # sentences = get_sentences(content)
    #
    # print(sentences)
