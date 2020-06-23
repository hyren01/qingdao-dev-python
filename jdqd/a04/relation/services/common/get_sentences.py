#!/usr/bin/env python
import re
from loguru import logger


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