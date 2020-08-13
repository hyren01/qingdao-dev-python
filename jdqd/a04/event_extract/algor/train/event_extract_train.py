#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年07月17
# ! -*- coding:utf-8 -*-
"""
 事件抽取模型训练
 使用条件批归一化进行向量融合
 对传入的数据进行处理，不再进行字符串的查找，而是精准传入下标，减小字符匹检索造成的问题
"""
import traceback
import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Average
from keras.models import Model
from keras.callbacks import Callback
from bert4keras.models import build_transformer_model
from bert4keras.backend import K, set_gelu
from bert4keras.optimizers import Adam
from bert4keras.layers import LayerNormalization
from jdqd.common.event_emm.model_utils import seq_gather, TOKENIZER, adversarial_training
from jdqd.a04.event_extract.algor.train.utils.event_extract_data_util import DataGenerator, get_data
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
import jdqd.a04.event_extract.config.ExtractTrainConfig as extract_train_config
import jdqd.common.event_emm.BertConfig as bert_config

# 设置激活函数
set_gelu('tanh')
# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)


def build_model():
    """
    调用模型参数，搭建事件抽取模型主体，先搭建触发词模型，然后围绕触发词下标搭建其他论元模型。
    :return: 各个论元模型对象
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构建bert模型主体
            bert_model = build_transformer_model(
                config_path=bert_config.config_path,
                checkpoint_path=bert_config.checkpoint_path,
                return_keras_model=False,
                model=bert_config.model_type
            )

            # l为模型内部的层名，格式为--str
            for l in bert_model.layers:
                bert_model.model.get_layer(l).trainable = True

            # 搭建模型
            # keras会自动对所有的占位张量添加batch_size维度
            # 动词输入 (batch_size, seq_len)
            trigger_start_in = Input(shape=(None,))
            trigger_end_in = Input(shape=(None,))
            # 动词下标输入 (batch_size, seq_len)
            trigger_index_start_in = Input(shape=(1,))
            trigger_index_end_in = Input(shape=(1,))
            # 宾语输入 (batch_size, seq_len)
            object_start_in = Input(shape=(None,))
            object_end_in = Input(shape=(None,))
            # 主语输入 (batch_size, seq_len)
            subject_start_in = Input(shape=(None,))
            subject_end_in = Input(shape=(None,))
            # 地点输入 (batch_size, seq_len)
            loc_start_in = Input(shape=(None,))
            loc_end_in = Input(shape=(None,))
            # 时间输入 (batch_size, seq_len)
            time_start_in = Input(shape=(None,))
            time_end_in = Input(shape=(None,))
            # 否定词输入 (batch_size, seq_len)
            negative_start_in = Input(shape=(None,))
            negative_end_in = Input(shape=(None,))

            # 将输入的占位符赋值给相应的变量(此处只是在使用时方便，没有其他的模型结构意义)
            # 动词输入
            trigger_start, trigger_end = trigger_start_in, trigger_end_in
            # 动词下标
            trigger_index_start, trigger_index_end = trigger_index_start_in, trigger_index_end_in
            # 宾语输入
            object_start, object_end = object_start_in, object_end_in
            # 主语输入
            subject_start, subject_end = subject_start_in, subject_end_in
            # 地点输入
            loc_start, loc_end = loc_start_in, loc_end_in
            # 时间输入
            time_start, time_end = time_start_in, time_end_in
            # 否定词输入
            negative_start, negative_end = negative_start_in, negative_end_in

            # bert_model.model.inputs为列表格式，含有两个张量，[token_ids(batch, seq_len), segment_ids(batch, seq_len)]
            # mask操作，将bert模型的输入的token_ids序列，进行维度扩充[batch_size, seq_len, 1],
            # 然后将初始填充为0的地方全部都用0代替，非0的地方都用1占位，
            # 这是为后边计算损失做准备，防止计算损失时，前期填充为0的位置也进行反向传播
            mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x[0], 2), 0), 'float32'))(bert_model.model.inputs)

            # 计算动词输出的起始终止标签，bert_model.model.output [batch_size, seq_len, 768]
            trigger_start_out = Dense(1, activation='sigmoid')(bert_model.model.output)
            trigger_end_out = Dense(1, activation='sigmoid')(bert_model.model.output)
            # 预测trigger动词的模型
            trigger_model = Model(bert_model.model.inputs, [trigger_start_out, trigger_end_out])

            # 将动词下标对应位置的子向量抽取出来并计算均值
            k1v = Lambda(seq_gather)([bert_model.model.output, trigger_index_start])
            k2v = Lambda(seq_gather)([bert_model.model.output, trigger_index_end])
            kv = Average()([k1v, k2v])
            # 融合动词词向量的句子张量，用来作为预测其它论元部分的向量
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
            # 否定词模型
            negative_model = Model(bert_model.model.inputs + [trigger_index_start_in, trigger_index_end_in],
                                   [negative_start_out, negative_end_out])

            # 主模型
            train_model = Model(
                bert_model.model.inputs + [trigger_start_in, trigger_end_in, trigger_index_start_in,
                                           trigger_index_end_in,
                                           object_start_in, object_end_in, subject_start_in, subject_end_in,
                                           loc_start_in,
                                           loc_end_in, time_start_in, time_end_in, negative_start_in, negative_end_in],
                [trigger_start_out, trigger_end_out, object_start_out, object_end_out, subject_start_out,
                 subject_end_out,
                 loc_start_out, loc_end_out, time_start_out, time_end_out, negative_start_out, negative_end_out])

            # 扩充维度, 构造成与mask矩阵相同的结构，方便后续计算模型各部分损失
            trigger_start = K.expand_dims(trigger_start, 2)
            trigger_end = K.expand_dims(trigger_end, 2)
            object_start = K.expand_dims(object_start, 2)
            object_end = K.expand_dims(object_end, 2)
            subject_start = K.expand_dims(subject_start, 2)
            subject_end = K.expand_dims(subject_end, 2)
            loc_start = K.expand_dims(loc_start, 2)
            loc_end = K.expand_dims(loc_end, 2)
            time_start = K.expand_dims(time_start, 2)
            time_end = K.expand_dims(time_end, 2)
            negative_start = K.expand_dims(negative_start, 2)
            negative_end = K.expand_dims(negative_end, 2)

            # 构造模型损失函数，使用mask矩阵将前期填充为0的位置全部掩掉，不进行反向传播。
            # 动词损失
            trigger_start_loss = K.binary_crossentropy(trigger_start, trigger_start_out)
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            trigger_start_loss = K.sum(trigger_start_loss * mask) / K.sum(mask)
            trigger_end_loss = K.binary_crossentropy(trigger_end, trigger_end_out)
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            trigger_end_loss = K.sum(trigger_end_loss * mask) / K.sum(mask)
            # 宾语损失
            object_start_loss = K.sum(K.binary_crossentropy(object_start, object_start_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            object_start_loss = K.sum(object_start_loss * mask) / K.sum(mask)
            object_end_loss = K.sum(K.binary_crossentropy(object_end, object_end_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            object_end_loss = K.sum(object_end_loss * mask) / K.sum(mask)
            # 主语损失
            subject_start_loss = K.sum(K.binary_crossentropy(subject_start, subject_start_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            subject_start_loss = K.sum(subject_start_loss * mask) / K.sum(mask)
            subject_end_loss = K.sum(K.binary_crossentropy(subject_end, subject_end_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            subject_end_loss = K.sum(subject_end_loss * mask) / K.sum(mask)
            # 地点损失
            loc_start_loss = K.sum(K.binary_crossentropy(loc_start, loc_start_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            loc_start_loss = K.sum(loc_start_loss * mask) / K.sum(mask)
            loc_end_loss = K.sum(K.binary_crossentropy(loc_end, loc_end_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            loc_end_loss = K.sum(loc_end_loss * mask) / K.sum(mask)
            # 时间损失
            time_start_loss = K.sum(K.binary_crossentropy(time_start, time_start_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            time_start_loss = K.sum(time_start_loss * mask) / K.sum(mask)
            time_end_loss = K.sum(K.binary_crossentropy(time_end, time_end_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            time_end_loss = K.sum(time_end_loss * mask) / K.sum(mask)
            # 否定词损失
            negative_start_loss = K.sum(K.binary_crossentropy(negative_start, negative_start_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            negative_start_loss = K.sum(negative_start_loss * mask) / K.sum(mask)
            negative_end_loss = K.sum(K.binary_crossentropy(negative_end, negative_end_out))
            # 使用mask矩阵，将前期填充为0的位置掩掉，不进行反向传播
            negative_end_loss = K.sum(negative_end_loss * mask) / K.sum(mask)

            # 合并损失
            loss = (trigger_start_loss + trigger_end_loss) + (object_start_loss + object_end_loss) + (
                    subject_start_loss + subject_end_loss) + (loc_start_loss + loc_end_loss) + (
                           time_start_loss + time_end_loss) + (negative_start_loss + negative_end_loss)

            train_model.add_loss(loss)
            train_model.compile(optimizer=Adam(extract_train_config.learning_rate))
            train_model.summary()

    return trigger_model, subject_model, object_model, time_model, loc_model, negative_model, train_model


def extract_items(text_in: str, maxlen, trigger_model, object_model, subject_model, loc_model, time_model,
                  negative_model):
    """
    传入待预测的文本字符串，调用各个模型，句子中的事件论元进行抽取，先抽取动词，然后使用动词下标和句子张量预测其它论元。
    :param text_in: （str）句子字符串
    :return: events(list)
                含有下标的事件各论元组成部分
                        [{ "time": [],
                            "loc": [],
                            "subject": [],
                            "trigger": [],
                            "object": [],
                            "privative": []}]
    :raise:type error
    """
    if not isinstance(text_in, str):
        logger.error("Type of text_in must be str!")
        raise TypeError

    # 对输入的句子进行分词
    _tokens = TOKENIZER.tokenize(text_in[:maxlen])
    # 对输入的句子进行ids化
    _text_token_ids, _text_segment_ids = TOKENIZER.encode(first_text=text_in[:maxlen])
    _text_token_ids, _text_segment_ids = np.array([_text_token_ids]), np.array([_text_segment_ids])
    _trigger_start_pre, _trigger_end_pre = trigger_model.predict([_text_token_ids, _text_segment_ids])
    _trigger_start_pre, _trigger_end_pre = np.where(_trigger_start_pre[0] > 0.5)[0], \
                                           np.where(_trigger_end_pre[0] > 0.4)[0]
    _triggers = []
    for i in _trigger_start_pre:
        j = _trigger_end_pre[_trigger_end_pre >= i]
        if len(j) > 0:
            j = j[0]
            _trigger = text_in[i - 1: j]
            _triggers.append((_trigger, i, j))
    # 根据触发词，抽取围绕触发词的各个组成部分
    if _triggers:
        events = []
        _text_token_ids = np.repeat(_text_token_ids, len(_triggers), 0)
        _text_segment_ids = np.repeat(_text_segment_ids, len(_triggers), 0)
        # 动词下标
        _trigger_start_index, _trigger_end_index = np.array([_s[1:] for _s in _triggers]).T.reshape((2, -1, 1))
        # 宾语
        _object_start_pre, _object_end_pre = object_model.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])
        # 主语
        _subject_start_pre, _subject_end_pre = subject_model.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])
        # 地点
        _loc_start_pre, _loc_end_pre = loc_model.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])
        # 时间
        _time_start_pre, _time_end_pre = time_model.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])
        # 否定词
        _negative_start_pre, _negative_end_pre = negative_model.predict(
            [_text_token_ids, _text_segment_ids, _trigger_start_index, _trigger_end_index])

        for i, _trigger in enumerate(_triggers):
            objects = []
            subjects = []
            locs = []
            times = []
            privatives = []
            # 宾语下标
            _object_start_index, _object_end_index = np.where(_object_start_pre[i] > 0.5)[0], \
                                                     np.where(_object_end_pre[i] > 0.4)[0]
            # 主语下标
            _subject_start_index, _subject_end_index = np.where(_subject_start_pre[i] > 0.5)[0], \
                                                       np.where(_subject_end_pre[i] > 0.4)[0]
            # 地点下标
            _loc_start_index, _loc_end_index = np.where(_loc_start_pre[i] > 0.5)[0], np.where(_loc_end_pre[i] > 0.4)[0]
            # 时间下标
            _time_start_index, _time_end_index = np.where(_time_start_pre[i] > 0.5)[0], \
                                                 np.where(_time_end_pre[i] > 0.4)[0]
            # 否定词下标
            _negative_start_index, _negative_end_index = np.where(_negative_start_pre[i] > 0.5)[0], \
                                                         np.where(_negative_end_pre[i] > 0.4)[0]

            # 开始抽取各个论元部分
            for i in _object_start_index:
                j = _object_end_index[_object_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _object = text_in[i - 1: j]
                    objects.append([_object, [str(i - 1), str(j)]])
            for i in _subject_start_index:
                j = _subject_end_index[_subject_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _subject = text_in[i - 1: j]
                    subjects.append([_subject, [str(i - 1), str(j)]])
            for i in _loc_start_index:
                j = _loc_end_index[_loc_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _loc = text_in[i - 1: j]
                    locs.append([_loc, [str(i - 1), str(j)]])
            for i in _time_start_index:
                j = _time_end_index[_time_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _time = text_in[i - 1: j]
                    times.append([_time, [str(i - 1), str(j)]])
            for i in _negative_start_index:
                j = _negative_end_index[_negative_end_index >= i]
                if len(j) > 0:
                    j = j[0]
                    _privative = text_in[i - 1: j]
                    privatives.append([_privative, [str(i - 1), str(j)]])

            events.append({
                "time": times,
                "loc": locs,
                "subject": subjects,
                "trigger": [[_trigger[0], [str(_trigger[1] - 1), str(_trigger[2])]]],
                "object": objects,
                "privative": privatives
            })

        return events
    else:
        return []


# def model_test(test_data):
#     """
#     传入测试数据，将测试结果保存到文件中
#     :param test_data: (list)测试数据
#     :return: None
#     :raise: TypeError
#     """
#     if not isinstance(test_data, list):
#         LOG.error("The type of test_data is wrong!")
#         raise TypeError
#
#     f = open(CONFIG.pred_path, 'w', encoding='utf-8')
#     f.write('[\n')
#     first = True
#     for d in iter(test_data):
#         if not first:
#             f.write(',\n')
#         else:
#             first = False
#         events = extract_items(d['sentence'])
#         s = json.dumps({
#             'sentence': d['sentence'],
#             'events': events
#         }, ensure_ascii=False)
#         f.write(s)
#     f.write(']\n')
#     f.close()


class Evaluate(Callback):
    """
    继承Callback类，改写内部方法,调整训练过程中的学习率，随着训练步数增加，逐渐减小学习率，根据指标保存最优模型。
    """

    def __init__(self, dev_data, maxlen, trained_model_path, trigger_model,
                 object_model, subject_model, loc_model, time_model, negative_model, train_model,
                 max_learning_rate=extract_train_config.learning_rate,
                 min_learning_rate=extract_train_config.min_learning_rate):
        Callback.__init__(self, )
        self.best = 0.
        self.passed = 0
        self.maxlen = maxlen
        self.dev_data = dev_data
        # 训练后模型路径
        self.trained_mdoel_path = trained_model_path
        # 学习率
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        # 论元模型
        self.trigger_model = trigger_model
        self.object_model = object_model
        self.subject_model = subject_model
        self.negative_model = negative_model
        self.time_model = time_model
        self.loc_model = loc_model
        # 主模型
        self.train_model = train_model

    def on_batch_begin(self, batch, logs=None):
        """
        在每个批次开始，判断批次步数，来调整训练时的学习率,第一个循环，学习率逐渐增大，
        第二个循环增至最大，随着训练步数的增加，学习率成线性减小。
        :param batch: 批次步数
        :param logs: 日志信息
        :return: None
        """
        # 当训练过的步数小于一个循环设定的步数时，学习率逐渐增大，首循环最后增至最大学习率
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * extract_train_config.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        # 当训练或的步数大于一个循环设定的步数且小于两倍设定步数时，也就是第二个循环学习率随步数线性减小至最小学习率
        # 在接下来的训练中一直保持最小学习率
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (self.max_learning_rate - self.min_learning_rate)
            lr += self.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        每个循环结束时判断指标是否最好，是则将模型保存下来。
        :param epoch: 循环次数
        :param logs: 日志信息
        :return: None
        """
        f1, precision, recall = self.evaluate()
        if f1 > self.best:
            self.best = f1
            self.train_model.save(self.trained_mdoel_path, include_optimizer=True)
        logger.info(f'f1: {f1}.4f, precision: {precision}.4f, recall: {recall}.4f, best f1: {self.best}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
        对传入的列表进行拉平
        :param lists: 传入二维列表
        :return: 拉平后的以后列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    def evaluate(self, ):
        """
        衡量元素将关系，计算模型在验证集上的f1\precision\recall
        :return: f1(float), precision(float), recall(float)
        """

        a, b, c = 1e-10, 1e-10, 1e-10

        for d in iter(self.dev_data):
            # 调用模型预测测试集中的事件
            extract_res = extract_items(d['sentence'], self.maxlen, self.trigger_model, self.object_model,
                                        self.subject_model, self.loc_model, self.time_model, self.negative_model)
            # 存储预测事件的集合
            _pred = set()
            # 存储真实事件的集合
            _real = set()
            # 遍历模型预测值，将事件论元连同触发词保存到元组中，
            # 构成围绕触发词的事件集合{(trigger, time),(trigger, loc),(trigger, subject), (trigger, object)}
            for event in extract_res:
                trigger = event['trigger'][0][0]
                for t in event['time']:
                    _pred.add((trigger, t[0]))
                for l in event['loc']:
                    _pred.add((trigger, l[0]))
                for s in event['subject']:
                    _pred.add((trigger, s[0]))
                for o in event['object']:
                    _pred.add((trigger, o[0]))
            # 遍历测试集真实值，同预测结果处理方式相同，将事件论元连同触发词保存到元组中
            # 构成围绕触发词的事件集合{(trigger, time),(trigger, loc),(trigger, subject), (trigger, object)}
            for event in d['events']:
                trigger = event['trigger'][0][0]
                for t in event['time']:
                    _real.add((trigger, t[0]))
                for l in event['loc']:
                    _real.add((trigger, l[0]))
                for s in event['subject']:
                    _real.add((trigger, s[0]))
                for o in event['object']:
                    _real.add((trigger, o[0]))

            # 真实事件和预测事件相同的个数
            a += len(_pred & _real)
            # 预测得到的事件个数
            b += len(_pred)
            # 真实事件个数
            c += len(_real)

        f1, precision, recall = 2 * a / (b + c), a / b, a / c

        return f1, precision, recall


def model_train(version, model_id, trained_model_dir="", data_dir="", all_steps=0, maxlen=160, epoch=40, batch_size=8,
                max_learning_rate=5e-5, min_learning_rate=1e-5, model_type="roberta"):
    """
    进行模型训练的主函数，搭建模型，加载模型数据，根据传入的参数进行模型训练
    :param version: 模型版本
    :param model_id: 模型编号
    :param data_dir: 补充数据的文件夹路径
    :param trained_model_dir: 训练后模型存放路径
    :param all_steps: 模型训练步数
    :param maxlen: 最大长度
    :param epoch: 循环次数
    :param batch_size: 批次大小
    :param max_learning_rate: 最大学习率
    :param min_learning_rate: 最小学习率
    :param version: 模型版本号
    :param model_id: 模型id
    :param model_type: 预训练模型类型，现在不用，后期会用到，
    :return: status, F1, precision, recall, corpus_num
    """
    try:
        # 补充数据文件夹更换
        if data_dir:
            extract_train_config.supplement_data_dir = data_dir
        # 判断训练后模型路径是否存在
        if not os.path.exists(trained_model_dir):
            os.makedirs(trained_model_dir)
        # 训练后模型路径
        trained_model_path = cat_path(trained_model_dir, f"extract_model_{version}_{model_id}")

        # 获取训练集、验证集
        train_data, dev_data = get_data(extract_train_config.train_data_path, extract_train_config.dev_data_path,
                                        data_dir)
        # 搭建模型
        trigger_model, object_model, subject_model, loc_model, time_model, negative_model, train_model = build_model()
        # 构造训练数据生成器
        train_d = DataGenerator(TOKENIZER, maxlen, train_data, batch_size)

        with SESS.as_default():
            with SESS.graph.as_default():
                # 构造callback模块的评估类
                evaluator = Evaluate(dev_data, maxlen, trained_model_path,
                                     trigger_model, object_model, subject_model, loc_model, time_model, negative_model,
                                     train_model, max_learning_rate, min_learning_rate)
                # 如果训练数据小于5000，则设置内部循环步数1000次为一个循环，否则集使用训练数据个数/batch_size计算得到的步数
                if all_steps:
                    pass
                elif len(train_data) < 5000:
                    all_steps = 1000
                else:
                    all_steps = train_d.__len__()

                # 构造对抗攻击训练
                adversarial_training(train_model, 'Embedding-Token', 0.1)

                # 模型训练
                train_model.fit_generator(train_d.__iter__(), steps_per_epoch=all_steps, epochs=epoch,
                                          callbacks=[evaluator])
                # 重载模型参数
                train_model.load_weights(trained_model_path)
                # 将验证集预测结果保存到文件中，暂时注释掉
                f1, precision, recall = evaluator.evaluate()
                logger.info(f"f1:{f1}, precision:{precision}, recall:{recall}")

        return {"status": "success", "version": version, "model_id": model_id,
                "results": {"f1": f1, "precison": precision, "recall": recall},
                "corpus_num": {"train_data": len(train_data), "dev_data": len(dev_data)}}
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        return {"status": "failed", "results": trace}


if __name__ == '__main__':
    # # 预测
    # with SESS.as_default():
    #     with SESS.graph.as_default():
    #         evaluator = Evaluate()
    #         TRAIN_MODEL.load_weights(extract_train_config.trained_model_path)
    #         f1, precision, recall = evaluator.evaluate()
    #         logger.info(f"f1:{f1}, precision:{precision}, recall:{recall}")

    model_train(version="0", model_id="0", trained_model_dir=extract_train_config.trained_model_dir,
                data_dir=extract_train_config.supplement_data_dir, maxlen=extract_train_config.maxlen,
                epoch=extract_train_config.epoch, batch_size=extract_train_config.batch_size,
                max_learning_rate=extract_train_config.learning_rate,
                min_learning_rate=extract_train_config.min_learning_rate, model_type="roberta")
