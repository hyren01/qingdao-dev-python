#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
加载spacy模型以及neuralcoref参数做英文指代消解
"""
import spacy
import neuralcoref
from feedwork.utils import logger


def get_spacy():
    """
    加载指代消解模型，返回模型对象。
    :return: nlp指代模型对象
    """
    # 加载spacy模型
    logger.info("开始加载spacy模型。。。")
    # spacy加载参数
    nlp = spacy.load('en_core_web_sm')
    # 加载字典，加载指代网络参数
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    # 构建管道，整合spacy和指代网络
    nlp.add_pipe(coref, name='neuralcoref')
    logger.info("spacy模型加载完成!")

    return nlp


def coref_data(nlp, line: str):
    """
    传入nlp模型和待消解的英文字符串，对英文文本进行指代消解，返回消解后的字符串。
    :param nlp: 指代消解模型对象
    :param line: (str)待消解的文本
    :return: res(str)消解后的文本
    """
    if not isinstance(line, str):
        logger.error("The type of content for coreference must be string!")
        raise TypeError
    # 调用模型传入待消解的文本
    doc = nlp(line)
    # 调用指代消解方法，得到指代消解后的文本内容
    res = doc._.coref_resolved

    return res


if __name__ == "__main__":
    data = "今晚国足进行比赛，他们信心满满，但是他们不一定能赢。"
    # res = execute(data)
    # print(res)
