#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
该模块加载事件抽取过程使用的所有模型，并返回模型对象，以及对数据预测值进行处理返回给flask模块
"""
import numpy as np
from pyhanlp import HanLP
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Average
from bert4keras.models import build_transformer_model
from bert4keras.layers import LayerNormalization
from jdqd.common.event_emm.data_utils import read_json
from jdqd.common.event_emm.model_utils import seq_gather, TOKENIZER
from feedwork.utils import logger
import jdqd.a04.event_extract.config.PredictConfig as pre_config
import jdqd.common.event_emm.BertConfig as bert_config

# 事件状态与id字典
STATE2ID = {'happened': 0, 'happening': 1, 'possible': 2}
ID2STATE = {0: 'happened', 1: 'happening', 2: 'possible'}
# 事件cameo字典
ID2CAMEO = read_json(pre_config.id2cameo_path)


# 加载事件抽取模型
def get_extract_model():
    """
    构建事件抽取模型结构，加载模型参数，返回模型对象
    1、使用bert输出预测动词下标
    2、使用bert输出融合动词下标预测事件时间、地点、主语、宾语、否定词
    :return: 各个部分的模型对象
    """
    # 构建bert模型主体
    bert_model = build_transformer_model(
        config_path=bert_config.config_path,
        return_keras_model=False,
        model=bert_config.model_type
    )

    # 搭建模型
    # 动词输入
    trigger_start_in = Input(shape=(None,))
    trigger_end_in = Input(shape=(None,))
    # 动词下标输入
    trigger_index_start_in = Input(shape=(1,))
    trigger_index_end_in = Input(shape=(1,))
    # 宾语输入
    object_start_in = Input(shape=(None,))
    object_end_in = Input(shape=(None,))
    # 主语输入
    subject_start_in = Input(shape=(None,))
    subject_end_in = Input(shape=(None,))
    # 地点输入
    loc_start_in = Input(shape=(None,))
    loc_end_in = Input(shape=(None,))
    # 时间输入
    time_start_in = Input(shape=(None,))
    time_end_in = Input(shape=(None,))
    # 否定词输入
    negative_start_in = Input(shape=(None,))
    negative_end_in = Input(shape=(None,))
    # 将模型外传入的下标赋值给模型内部变量(只是为了将模型中应用与构建Model的输入区分开来)
    trigger_index_start, trigger_index_end = trigger_index_start_in, trigger_index_end_in

    trigger_start_out = Dense(1, activation='sigmoid')(bert_model.model.output)
    trigger_end_out = Dense(1, activation='sigmoid')(bert_model.model.output)
    # 预测trigger动词的模型
    trigger_model = Model(bert_model.model.inputs, [trigger_start_out, trigger_end_out])

    # 按照动词下标采集字向量
    k1v = Lambda(seq_gather)([bert_model.model.output, trigger_index_start])
    k2v = Lambda(seq_gather)([bert_model.model.output, trigger_index_end])
    kv = Average()([k1v, k2v])
    # 使用归一化融合动词词向量与句子张量
    t = LayerNormalization(conditional=True)([bert_model.model.output, kv])

    # 宾语模型输出
    object_start_out = Dense(1, activation='sigmoid')(t)
    object_end_out = Dense(1, activation='sigmoid')(t)
    # 主语模型输出
    subject_start_out = Dense(1, activation='sigmoid')(t)
    subject_end_out = Dense(1, activation='sigmoid')(t)
    # 地点模型输出
    loc_start_out = Dense(1, activation='sigmoid')(t)
    loc_end_out = Dense(1, activation='sigmoid')(t)
    # 时间模型输出
    time_start_out = Dense(1, activation='sigmoid')(t)
    time_end_out = Dense(1, activation='sigmoid')(t)
    # 否定词模型输出
    negative_start_out = Dense(1, activation='sigmoid')(t)
    negative_end_out = Dense(1, activation='sigmoid')(t)

    # 输入text和trigger，预测object
    object_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                         [object_start_out, object_end_out])
    # 输入text和trigger，预测subject
    subject_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                          [subject_start_out, subject_end_out])
    # 输入text和trigger，预测loc
    loc_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                      [loc_start_out, loc_end_out])
    # 输入text和trigger，预测time
    time_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                       [time_start_out, time_end_out])
    # 输入text和trigger，预测否定词negative
    negative_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                           [negative_start_out, negative_end_out])

    # 主模型
    train_model = Model(
        bert_model.model.inputs + [trigger_start_in, trigger_end_in, trigger_index_start_in, trigger_index_end_in,
                                   object_start_in, object_end_in, subject_start_in, subject_end_in, loc_start_in,
                                   loc_end_in, time_start_in, time_end_in, negative_start_in, negative_end_in],
        [trigger_start_out, trigger_end_out, object_start_out, object_end_out, subject_start_out, subject_end_out,
         loc_start_out, loc_end_out, time_start_out, time_end_out, negative_start_out, negative_end_out])
    # 加载事件抽取模型参数
    train_model.load_weights(pre_config.event_extract_model_path)

    return trigger_model, object_model, subject_model, loc_model, time_model, negative_model


# 加载事件时态判断模型
def get_state_model():
    """
    构建事件状态模型，加载模型参数，返回模型对象
    使用bert输出融合动词下标预测事件状态
    :return: state_model
    """
    # 构建bert模型主体
    bert_model = build_transformer_model(
        config_path=bert_config.config_path,
        return_keras_model=False,
        model=bert_config.model_type
    )
    # 动词下标输入
    trigger_start_index = Input(shape=(1,))
    trigger_end_index = Input(shape=(1,))
    # 获取动词向量
    k1v = Lambda(seq_gather)([bert_model.model.output, trigger_start_index])
    k2v = Lambda(seq_gather)([bert_model.model.output, trigger_end_index])
    kv = Average()([k1v, k2v])
    # 将动词向量与bert模型输出也就是句子张量进行融合
    t = LayerNormalization(conditional=True)([bert_model.model.output, kv])
    # 取出[CLS]对应的向量用来做分类
    t = Lambda(lambda x: x[:, 0])(t)
    # 预测事件状态
    state_out_put = Dense(3, activation='softmax')(t)
    # 主模型
    state_model = Model(bert_model.model.inputs + [trigger_start_index, trigger_end_index], state_out_put)

    # 加载模型
    state_model.load_weights(pre_config.event_state_model_path)

    return state_model


def get_cameo_model():
    """
    加载事件类别（CAMEO）模型，返回事件类别模型对象
    使用bert模型输出融合动词下标，预测事件cameo
    :return: 事件cameo模型
    """
    # 搭建bert模型主体
    bert_model = build_transformer_model(
        config_path=bert_config.config_path,
        return_keras_model=False,
        model=bert_config.model_type
    )
    # 取出[CLS]对应的向量用来做分类
    t = Lambda(lambda x: x[:, 0])(bert_model.model.output)
    # 预测事件cameo
    cameo_out_put = Dense(len(ID2CAMEO), activation='softmax')(t)
    # cameo模型
    cameo_model = Model(bert_model.model.inputs, cameo_out_put)
    # 加载模型参数
    cameo_model.load_weights(pre_config.event_cameo_model_path)

    return cameo_model


def extract_items(text_in: str, state_model, trigger_model, object_model, subject_model, loc_model, time_model,
                  negative_model):
    """
    传入待抽取事件的句子，抽取事件的各个模型，对事件句子中的事件进行抽取。
    :param text_in: (str)待抽取事件的句子
    :param state_model: 事件状态模型
    :param trigger_model: 事件触发词模型
    :param object_model: 事件宾语模型
    :param subject_model: 事件主语模型
    :param loc_model: 事件地点模型
    :param time_model: 事件时间模型
    :param negative_model: 事件否定词模型
    :return: events(list)事件列表
    :raise:TypeError 传入的待抽取的句子是否为字符串
    """
    if not isinstance(text_in, str):
        logger.error("待抽取的句子不是字符串！")
        raise TypeError
    # 使用bert分字器对字符串进行分字并ids化
    token_ids, segment_ids = TOKENIZER.encode(first_text=text_in[:pre_config.maxlen])
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    # 动词预测值
    trigger_pred_start, trigger_pred_end = trigger_model.predict([token_ids, segment_ids])
    # 分别设置动词起始下标和终止下标置信度为0.5和0.4，获取动词下标
    trigger_start_index, trigger_end_index = np.where(trigger_pred_start[0] > 0.5)[0], np.where(trigger_pred_end[0] > 0.4)[0]

    # 创建动词列表，保存(动词，动词起始下标， 动词终止下标， 事件状态)
    _triggers = []

    # 遍历动词起始下标，查找终止下标候选集，选取第一个终止下标作为匹配对切分动词
    for i in trigger_start_index:
        # 获取动词终止下标候选集
        j = trigger_end_index[trigger_end_index >= i]

        # 判断候选集中是否为空，如果不为空则说明存在动词
        if len(j) > 0:
            # 选取候选集第一个下标作为动词终止下标
            j = j[0]
            # 获取动词
            _trigger = text_in[i - 1: j]
            # 将句子编码序列和预测得到的动词下标输入到事件状态模型，预测事件状态[[0.01,0.10,0.89]]
            state = state_model.predict([token_ids, segment_ids, np.array([i], dtype=np.int32), np.array([j], dtype=np.int32)])
            # 获取置信度最大的下标，也就是状态id
            state = np.argmax(state, axis=1)
            # 按照id查找字典，获取状态happened happening possible
            state = ID2STATE[state[0]]
            # 将动词、动词下标、状态组成的元组添加到动词列表中
            _triggers.append((_trigger, i, j, state))

    # 如果动词不为空，则说明句子中存在事件
    if _triggers:
        # 事件列表
        events = []
        # 构造预测事件论元的字符串编码集，将句子ids编码序列和segment编码序列在沿着0轴复制n次，n为动词个数，
        # 也就是一个动词对应一个句子输入
        batch_token_ids = np.repeat(token_ids, len(_triggers), 0)
        batch_segment_ids = np.repeat(segment_ids, len(_triggers), 0)
        # 动词下标
        trigger_start, trigger_end = np.array([_s[1:3] for _s in _triggers]).T.reshape((2, -1, 1))
        # 宾语预测值
        object_pred_start, object_pred_end = object_model.predict([batch_token_ids, batch_segment_ids, trigger_start, trigger_end])
        # 主语预测值
        subject_pred_start, subject_pred_end = subject_model.predict([batch_token_ids, batch_segment_ids, trigger_start, trigger_end])
        # 地点预测值
        loc_pred_start, loc_pred_end = loc_model.predict([batch_token_ids, batch_segment_ids, trigger_start, trigger_end])
        # 时间预测值
        time_pred_start, time_pred_end = time_model.predict([batch_token_ids, batch_segment_ids, trigger_start, trigger_end])
        # 否定词预测值
        negative_pred_start, negative_pred_end = negative_model.predict([batch_token_ids, batch_segment_ids, trigger_start, trigger_end])

        # 遍历动词，根据前面预测得到的论元预测值，获取对应于动词的其他各个论元组成部分
        for i, _trigger in enumerate(_triggers):
            objects = []
            subjects = []
            locs = []
            times = []
            negatives = []
            # 宾语下标
            object_start_index, object_end_index = np.where(object_pred_start[i] > 0.5)[0], np.where(object_pred_end[i] > 0.4)[0]
            # 主语下标
            subject_start_index, subject_end_index = np.where(subject_pred_start[i] > 0.5)[0], np.where(subject_pred_end[i] > 0.4)[0]
            # 地点下标
            loc_start_index, loc_end_index = np.where(loc_pred_start[i] > 0.5)[0], np.where(loc_pred_end[i] > 0.4)[0]
            # 时间下标
            time_start_index, time_end_index = np.where(time_pred_start[i] > 0.5)[0], np.where(time_pred_end[i] > 0.4)[0]
            # 否定词下标
            negative_start_index, negative_end_index = np.where(negative_pred_start[i] > 0.5)[0], np.where(negative_pred_end[i] > 0.4)[0]

            # 按照下标，在字符串中索引对应的论元，（方法同动词）
            # 获取宾语
            for i in object_start_index:
                j = object_end_index[object_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _object = text_in[i - 1: j]
                    objects.append(_object)
            # 获取主语
            for i in subject_start_index:
                j = subject_end_index[subject_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _subject = text_in[i - 1: j]
                    subjects.append(_subject)
            # 获取地点
            for i in loc_start_index:
                j = loc_end_index[loc_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _loc = text_in[i - 1: j]
                    locs.append(_loc)
            # 获取时间
            for i in time_start_index:
                j = time_end_index[time_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _time = text_in[i - 1: j]
                    times.append(_time)
            # 获取否定词
            for i in negative_start_index:
                j = negative_end_index[negative_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _negative = text_in[i - 1: j]
                    negatives.append(_negative)

            # 将单个事件所有论元保存到字典中并添加入事件列表中
            events.append({
                "event_datetime": times[0] if times else "",
                "event_location": locs[0] if locs else "",
                "subject": ",".join(subjects) if subjects else "",
                "verb": _trigger[0],
                "object": ",".join(objects) if objects else "",
                "negative_word": negatives[0] if negatives else "",
                "state": _trigger[3],
                "triggerloc_index": [int(_trigger[1] - 1), int(_trigger[2] - 1)]
            })
        # {'event_datetime': "", 'event_location': "", 'subject': '美国',
        # 'verb': '采取', 'object': '数十次由总统承诺停止的联合演习',
        # 'negative_word': "取消", 'state': 'possible', triggerloc_index:[12,15]}
        return events
    else:
        return []


def get_ners(events: list):
    """
    传入事件列表，将事件论元拼接为字符串，使用hanlp抽取事件中的实体名词，补充到事件列表中
    :param events: (list)事件列表
    :return: events (list)补充实体成分后的事件字典列表
    :raise:TypeError 事件列表类型错误
    """
    if not isinstance(events, list):
        logger.error("用于抽取实体的事件列表类型错误！")
        raise TypeError

    # 遍历事件列表
    for event in events:
        # 获取事件所有论元，拼接成事件句子
        sentence = ''.join([event[i] for i in event.keys() if i != 'state' and i != 'triggerloc_index'])
        # 使用hanlp对事件句子进行实体识别
        words = HanLP.segment(sentence)
        # 在事件字典中加入namedentity
        event["namedentity"] = {'organization': [], 'location': [], 'person': []}

        # 遍历实体识别后的词
        for once in words:
            # 判断是否为地点实体
            if str(once.nature).startswith("ns"):
                if str(once.word) in event["namedentity"]["location"]:
                    pass
                else:
                    event["namedentity"]["location"].append(str(once.word))
            # 判断是否为人物实体
            elif str(once.nature).startswith("nr"):
                if str(once.word) in event["namedentity"]["person"]:
                    pass
                else:
                    event["namedentity"]["person"].append(str(once.word))
            # 判断是否为机构实体
            elif str(once.nature).startswith("nt"):
                if str(once.word) in event["namedentity"]["organization"]:
                    pass
                else:
                    event["namedentity"]["organization"].append(str(once.word))

    return events


def get_event_cameo(cameo_model, events: list):
    """
    传入事件cameo模型和事件列表，判断每个事件的cameo，并添加到事件列表中
    :param cameo_model:事件cameo模型
    :param events:(list) 事件列表
    :return:events(list)事件列表
    :raise:TypeError 事件列表类型
    """
    if not isinstance(events, list):
        logger.error("用于预测事件cameo的事件列表类型错误！")
        raise TypeError

    # 遍历事件列表
    for event in events:
        # 使用将主、谓、宾、否定词拼接为事件短句，预测事件cameo
        short_sentence = "".join([event["subject"], event["negative_word"], event["verb"], event["object"]])
        # 使用bert分字器对事件短句进行分字并ids化
        token_ids, segment_ids = TOKENIZER.encode(first_text=short_sentence)
        # 转化为数组，模型预测需要ndarray
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        # 获取cameo预测值 [[0.001,0.002,0.005,0.2,0.100,0.51]]
        event_cameo = cameo_model.predict([token_ids, segment_ids])
        # 获取cameo的id [cameo_id]
        event_cameo = np.argmax(event_cameo, axis=1)
        event_id = event_cameo[0]
        # 根据cameo_id获取cameo
        event["cameo"] = ID2CAMEO[f"{event_id}"]

    return events


if __name__ == '__main__':
    state_model = get_state_model()
    trigger_model, object_model, subject_model, loc_model, time_model, privative_model = get_extract_model()
    cameo_model = get_cameo_model()
    # while True:
    #     sentence = input("请输入想要预测的句子。")
    #     R = extract_items(sentence, state_model, trigger_model, object_model, subject_model, loc_model, time_model,
    #                       privative_model)
    #     print(R)
    #
    #     a = get_event_cameo(cameo_model, R)
