#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
调用小牛翻译接口实现中英互译
"""
import json
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError
from feedwork.utils import logger
import jdqd.a04.event_extract.config.PredictConfig as pre_config


def transform_any_2_en(article: str):
    """
    调用小牛翻译接口，将任意语言的文章翻译为英文
    :param article: (str)文章
    :return: content(str)英文文章
    :raise:TypeError
    """
    if not isinstance(article, str):
        logger.error("待翻译为英文的内容格式错误，需要字符串格式！")
        raise TypeError

    # 待编码的数据
    data = {"from": "auto", "to": "en", "apikey": pre_config.user_key, "src_text": article}

    try:
        # 编码请求数据
        data_en = urlencode(data)
        req = f"{pre_config.translate_url}&{data_en}"
        # 访问小牛发起翻译请求
        res = urlopen(req)
        # 获取请求反馈
        res = res.read()
        # 解析反馈结果
        res_dict = json.loads(res)

        # 判断小牛是否已经翻译，将反馈结果赋值于content
        if "tgt_text" in res_dict:
            content = res_dict['tgt_text']
        else:
            content = res

        return content

    except HTTPError:
        logger.error('翻译时发生的错误，通常是http请求太大')
        logger.error(str(HTTPError))
        # 请求失败返回空字符串
        return ''


def transform_any_2_zh(article: str):
    """
    将任意语言的文章翻译为中文
    :param article: (str)文章
    :return: content(str)中文文章
    :raise:TypeError
    """
    if not isinstance(article, str):
        logger.error("待翻译为中文的内容格式错误，需要字符串格式！")
        raise TypeError

    # 待编码的数据
    data = {"from": "auto", "to": "zh", "apikey": pre_config.user_key, "src_text": article}
    try:
        # 编码请求数据
        data_en = urlencode(data)
        req = f"{pre_config.translate_url}&{data_en}"
        # 访问小牛发起翻译请求
        res = urlopen(req)
        # 获取请求反馈
        res = res.read()
        # 解析反馈结果
        res_dict = json.loads(res)
        # 判断小牛是否已经翻译，将反馈结果赋值于content
        if "tgt_text" in res_dict:
            content = res_dict['tgt_text']
        else:
            content = res

        return content
    except HTTPError:
        logger.error('翻译时发生的错误，通常是http请求太大')
        logger.error(str(HTTPError))
        # 请求失败返回空字符串
        return ''


if __name__ == "__main__":
    article = "小明给在山区支教的同学打了一个电话。"
    trans = transform_any_2_en(article)
