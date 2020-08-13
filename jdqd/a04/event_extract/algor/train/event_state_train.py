#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件状态模型训练模块
"""
import numpy as np
import tensorflow as tf
from keras.layers import Input, Average, Lambda, Dense
from keras.models import Model
from keras.callbacks import Callback
from bert4keras.models import build_transformer_model
from bert4keras.backend import K
from bert4keras.layers import LayerNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from feedwork.utils import logger
from jdqd.common.event_emm.model_utils import seq_gather, TOKENIZER
from jdqd.a04.event_extract.algor.train.utils.event_state_data_util import DataGenerator, get_data
import jdqd.a04.event_extract.config.StateTrainConfig as state_train_config
import jdqd.common.event_emm.BertConfig as bert_config

# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = state_train_config.trained_model_path
# 加载训练集和验证集
TRAIN_DATA, DEV_DATA = get_data(state_train_config.train_data_path, state_train_config.dev_data_path,
                                state_train_config.supplement_data_dir)
# 状态标签字典
ID2STATE = state_train_config.id2state
STATE2ID = state_train_config.state2id


def build_model():
    """
    搭建模型主体。
    :return: 模型对象
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

            # 动词起始，终止下标，keras会自动补充batch这一维度
            # [batch_size, 1]
            trigger_start_index = Input(shape=(1,))
            trigger_end_index = Input(shape=(1,))

            # 将动词下标对应位置的子向量抽取出来并计算均值
            k1v = Lambda(seq_gather)([bert_model.model.output, trigger_start_index])
            k2v = Lambda(seq_gather)([bert_model.model.output, trigger_end_index])
            kv = Average()([k1v, k2v])
            # 融合动词词向量的句子张量
            t = LayerNormalization(conditional=True)([bert_model.model.output, kv])
            # 取出[CLS]对应的向量用来做分类
            t = Lambda(lambda x: x[:, 0])(t)
            # 预测事件状态
            state_out_put = Dense(3, activation='softmax')(t)
            # 构建状态预测模型
            state_model = Model(bert_model.model.inputs + [trigger_start_index, trigger_end_index], state_out_put)
            # 设置学习率、优化器、损失函数以及每个批次的评估指标
            state_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=Adam(state_train_config.learning_rate),
                                metrics=['accuracy'])

            state_model.summary()

    return state_model


# 构建模型
MODEL = build_model()
# 构建训练集以及测试集生成类
TRAIN_D = DataGenerator(TOKENIZER, state_train_config.maxlen, TRAIN_DATA, STATE2ID, state_train_config.batch_size)


def extract_items():
    """
    调用训练的模型对测试集进行预测,将预测结果写入文件中
    :return: None
    """
    # 预测结果
    result = []
    for once in DEV_DATA:
        # 防止句子过长，使用最大长度进行切分
        text = once["sentence"][:state_train_config.maxlen]
        # 对句子进行编码
        text_token_ids, text_segment_ids = TOKENIZER.encode(first_text=text)
        text_token_ids, text_segment_ids = np.array([text_token_ids]), np.array([text_segment_ids])

        for event in once["events"]:
            # 获取动词下标
            trigger_start_index, trigger_end_index = int(event["trigger"][0][1][0]) + 1, int(event["trigger"][0][1][1])
            trigger_start_index, trigger_end_index = np.array([trigger_start_index], dtype=np.int32), np.array(
                [trigger_end_index], dtype=np.int32)
            # 预测触发词状态[[0.1,0.001,0.4]]
            pre = MODEL.predict([text_token_ids, text_segment_ids, trigger_start_index, trigger_end_index])
            # [3]
            pre = np.argmax(pre, axis=1)
            # 将预测结果添加到result中
            result.append(pre[0])

    return result


class Evaluate(Callback):
    """
    继承Callback类，改写内部方法,调整训练过程中的学习率，随着训练步数增加，逐渐减小学习率，根据指标保存最优模型。
    """

    def __init__(self):
        Callback.__init__(self, )
        self.best = 0.
        self.passed = 0

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
            lr = (self.passed + 1.) / self.params['steps'] * state_train_config.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        # 当训练或的步数大于一个循环设定的步数且小于两倍设定步数时，也就是第二个循环学习率随步数线性减小至最小学习率
        # 在接下来的训练中一直保持最小学习率
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (
                        state_train_config.learning_rate - state_train_config.min_learning_rate)
            lr += state_train_config.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
         每个循环结束时判断指标是否最好，是则将模型保存下来
         :param epoch: (int)循环次数
         :param logs: 日志信息
         :return: None
        """
        accuracy, f1, precision, recall = self.evaluate()
        if accuracy > self.best:
            self.best = accuracy
            MODEL.save(TRAINED_MODEL_PATH, include_optimizer=True)
        logger.info(f'accuracy: {accuracy}.4f, best accuracy: {self.best}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
        对传入的列表进行拉平
        :param lists: (list)传入二维列表
        :return: 拉平后的列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    @staticmethod
    def evaluate():
        """
        评估模型效果
        :return: 准确率
        """
        # 验证集预测值
        y_pred = extract_items()
        # 验证集真实值
        y_true = [STATE2ID[event["state"]] for once in DEV_DATA for event in once["events"]]
        # 计算验证集准确率
        accuracy = accuracy_score(y_true, y_pred)
        f1, precision, recall, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

        return accuracy, f1, precision, recall


def model_train():
    """
    进行模型训练。
    :return: None
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构造评估对象
            evaluator = Evaluate()
            # 模型训练
            MODEL.fit_generator(TRAIN_D.__iter__(), steps_per_epoch=TRAIN_D.__len__(), epochs=40, callbacks=[evaluator])
            # 模型参数重载
            MODEL.load_weights(TRAINED_MODEL_PATH)
            accuracy, f1, precision, recall = evaluator.evaluate()
            logger.info(f"accuracy:{accuracy}")
            return accuracy, f1, precision, recall


if __name__ == '__main__':
    model_train()

    # 查看验证集预测结果
    # with SESS.as_default():
    #     with SESS.graph.as_default():
    #         MODEL.load_weights(TRAINED_MODEL_PATH)
    #
    #         pred = extract_items()
    #         i = 0
    #         for once in DEV_DATA:
    #             for event in once["events"]:
    #                 real = event["state"]
    #                 predict = ID2STATE[pred[i]]
    #                 event["state"] = f"{real}/{predict}"
    #                 i += 1
    #
    #         import json
    #         data = json.dumps(DEV_DATA, ensure_ascii=False, indent=4)
    #         with open("./predict.json", "w", encoding="utf-8") as file:
    #             file.write(data)
