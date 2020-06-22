#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
提供事件匹配模块模型加载、模型预测、预测结果的排序整理，供flask模块调用
"""
import numpy as np
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense
from keras.models import Model
from jdqd.common.event_emm.model_utils import TOKENIZER
from jdqd.common.event_emm.data_utils import data_process, get_sentences, file_reader
from jdqd.a01.event_match.algor.common.utils import DataGenerator, supplement
from jdqd.a01.event_match.algor.predict.get_abstract import get_abstract
from feedwork.utils import logger
import jdqd.a01.event_match.config.PredictConfig as pre_config
import jdqd.common.event_emm.BertConfig as bert_config


def load_match_model():
    """
    加载模型，返回模型对象
    :return: model
    """
    # 构建bert模型
    bert_model = build_transformer_model(config_path=bert_config.config_path,
                                         model=bert_config.model_type, return_keras_model=False)

    # 构建模型主体
    # 取出[CLS]对应的向量用来做分类
    t = Lambda(lambda x: x[:, 0])(bert_model.model.output)
    # 模型预测输出
    output = Dense(units=2,
                   activation='softmax')(t)

    model = Model(bert_model.model.inputs, output)
    model.summary()

    logger.info("开始加载匹配模型。。。")
    model.load_weights(pre_config.match_model_path)
    logger.info("匹配模型加载完成！")

    return model


def generate_samples(event_list: list, sentences: list):
    """
    传入事件列表和语句列表，生成匹配对进行预测
    :param event_list: (list)事件列表
    :param sentences: (list)待匹配的句子列表
    :return: samples(list)样本列表  [(event, sentence, 0), ]
    """
    samples = []
    for once in sentences:
        if once:
            for event in event_list:
                if event:
                    samples.append((event[-1], once, str(0)))
    return samples


def get_events():
    """
    获取事件列表文件中的所有事件，返回解析好的事件列表
    :return: event_list(list)事件列表[(event_id, event), ]
    """
    # 获取事件列表中的事件
    content = file_reader(pre_config.allevent_path)
    events = content.split('\n')
    event_list = []
    for once in events:
        if once:
            event_id = once.split('`')[0]
            event = once.split('`')[1].replace(' ', '')
            event_list.append((event_id, event))

    return event_list


def get_parts(content: str):
    """
    传入文章字符串，获取文章的头尾部分。
    :param content: (str)文章内容
    :return: parts(list)文章头尾部分
    """
    parts = []

    if len(content) > 128:
        parts.append(content[0:128])
        parts.append(content[-128:])
    else:
        parts.append(content)

    return parts


def sort_socres(event_list: list, pred: list):
    """
    传入事件列表和预测结果，将预测结果按照相似度进行排序
    :param event_list: (list)事件列表
    :param pred: (list)
    :return: event_sorted(list)排序后的预测结果
    """
    predicted_event = {}
    event_scores = {}
    # {event:[event_id, []]}
    for key in event_list:
        predicted_event[key[1]] = [key[0], []]
    # {event:[event_id, [score, ]]}
    for once in pred:
        predicted_event[once[0]][1].append(once[-1])

    # 遍历事件预测值字典，每个事件取最大相似度 {event_id:score}
    for i in predicted_event:
        event_scores[predicted_event[i][0]] = max(predicted_event[i][1])
    # 按照相似度将列表化后的事件相似度字典进行排序
    event_sorted = list(sorted(event_scores.items(), key=lambda e: e[1], reverse=True))
    # 遍历列表，将事件id 和相似度按照字典格式保存到列表中，[{event_id: , ratio： }， ]
    event_sorted = [{'event_id': elem[0], 'ratio': elem[1]} for elem in event_sorted]

    return event_sorted


def get_predict_result(model, event_list: list, title: str, content: str, sample_type: str):
    """
    传入模型对象以及事件列表和待匹配的文本，返回匹配后的事件id及对应的相似度
    :param model: 匹配模型对象
    :param event_list: (list)事件列表
    :param title: (str)文章标题
    :param content: (str)文章内容
    :param sample_type: (str)匹配类型
    :return: {事件ID:score}
    """

    def evaluate(data: iter):
        """
        传入经过ids化的数据，进行预测
        :param data: (iter) ids化后的数据
        :return: results(list)相似度列表
        """
        results = []
        for x_true, y_true in data:
            # 调用模型，预测样本相似度[batch, 2]
            y_pred = model.predict(x_true)
            # 获取1维度的分数作为相似度，并重新reshape为一行，转化为列表
            results.extend(np.reshape(y_pred[:, 1], (-1,)).tolist())

        return results

    # 清洗标题
    title = data_process(title)
    title = supplement(title)
    # 清洗文章内容
    content = data_process(content)
    content = supplement(content)
    # 获取以标题构造的样本[(event, title),(event, title),(event, title)]
    title_samples = generate_samples(event_list, [title])

    # 判断文章样本类型，根据类型生成样本
    if sample_type == 'abstract':
        # 抽取文章摘要句子
        summary_sentences = get_abstract(content)
        content_samples = generate_samples(event_list, summary_sentences)
    elif sample_type == "all":
        # 使用文章中的每个句子做匹配对
        sentences = get_sentences(content)
        content_samples = generate_samples(event_list, sentences)
    else:
        # 获取文章的部分片段作为样本
        parts_sentences = get_parts(content)
        content_samples = generate_samples(event_list, parts_sentences)

    # 标题样本生成对象
    title_generator = DataGenerator(title_samples, TOKENIZER, max_length=pre_config.maxlen, batch_size=pre_config.batch_size)
    # 获取文章标题匹配结果
    title_results = evaluate(title_generator)
    # 文章内容样本生成对象
    content_generator = DataGenerator(content_samples, TOKENIZER, max_length=pre_config.maxlen,
                                      batch_size=pre_config.batch_size)
    # 获取文章内容匹配结果
    content_results = evaluate(content_generator)

    # 整理匹配结果
    # [[event1, title, score], ]
    title_predicted = []
    for elem, pred in zip(title_generator.data, title_results):
        title_predicted.append([elem[0], elem[1], pred])

    # [[event1, sentence, score], ]
    content_predicted = []
    for elem, pred in zip(content_generator.data, content_results):
        content_predicted.append([elem[0], elem[1], pred])

    # 标题匹配结果排序
    title_pred = sort_socres(event_list, title_predicted)
    # 文章内容匹配结果排序
    content_pred = sort_socres(event_list, content_predicted)

    return title_pred, content_pred


if __name__ == '__main__':
    content = '美韩决定举行大规模联合军演'
    title = '韩美取消联合军演'
