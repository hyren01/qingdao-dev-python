#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件匹配模型训练模块
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score
from bert4keras.backend import K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.layers import Dense, Dropout, Lambda
from keras.models import Model
from keras.callbacks import Callback
from jdqd.a01.event_match.algor.common.utils import load_data, DataGenerator
from jdqd.common.event_emm.model_utils import TOKENIZER
from feedwork.utils import logger
import jdqd.a01.event_match.config.MatchTrainConfig as match_train_config
import jdqd.common.event_emm.BertConfig as bert_config

# 判断是否需要进行初次训练或者二次训练
if not os.path.exists(match_train_config.first_trained_model_path):
    logger.info("加载首次训练路径。。。")
    TRAINED_MODEL_PATH = match_train_config.first_trained_model_path
    TRAIN_DATA_PATH = match_train_config.first_train_data_path
    DEV_DATA_PATH = match_train_config.first_dev_data_path
    TEST_DATA_PATH = match_train_config.first_test_data_path
else:
    logger.info("加载二次训练路径。。。")
    TRAINED_MODEL_PATH = match_train_config.second_trained_model_path
    TRAIN_DATA_PATH = match_train_config.second_train_data_path
    DEV_DATA_PATH = match_train_config.second_dev_data_path
    TEST_DATA_PATH = match_train_config.second_test_data_path

# 加载数据
TRAIN_DATA = load_data(TRAIN_DATA_PATH)
DEV_DATA = load_data(DEV_DATA_PATH)
TEST_DATA = load_data(TEST_DATA_PATH)
# 数据生成器
TRAIN_DATAGENERATOR = DataGenerator(TRAIN_DATA, TOKENIZER, max_length=match_train_config.maxlen, batch_size=match_train_config.batch_size)
DEV_DATAGENERATOR = DataGenerator(DEV_DATA, TOKENIZER, max_length=match_train_config.maxlen, batch_size=match_train_config.batch_size)
TEST_DATAGENERATOR = DataGenerator(TEST_DATA, TOKENIZER, max_length=match_train_config.maxlen, batch_size=match_train_config.batch_size)


def build_model():
    """
    搭建模型结构，返回模型对象
    :return: model
    """
    # 构建bert模型
    bert_model = build_transformer_model(config_path=bert_config.config_path, checkpoint_path=bert_config.checkpoint_path,
                                         model=bert_config.model_type, return_keras_model=False)
    # l为模型内部的层名，格式为--str
    for l in bert_model.layers:
        bert_model.model.get_layer(l).trainable = True

    # 构建模型主体
    t = Lambda(lambda x: x[:, 0])(bert_model.model.output)  # 取出[CLS]对应的向量用来做分类
    t = Dropout(match_train_config.drop_out_rate)(t)
    # 模型预测输出
    output = Dense(units=2,
                   activation='softmax')(t)

    model = Model(bert_model.model.inputs, output)
    model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(match_train_config.learning_rate),  # 用足够小的学习率
        metrics=['accuracy'],
    )

    return model


# 构建模型
MATCH_MODEL = build_model()


def evaluate(data):
    """
    对传入的数据进行预测，并评估模型效果
    :param data: (iter) 传入测试数据[batch_token_ids, batch_segment_ids], batch_labels
    :return: 评估指标准确率
    """
    y_pred_total = []
    y_true_total = []
    for x_true, y_true in data:
        # 预测值最大的维度就是相似度标签，模型预测值为[[0.1, 0.9]]
        y_pred = MATCH_MODEL.predict(x_true).argmax(axis=1)
        # 将预测值转化为列表保存
        y_pred_total.extend(np.reshape(y_pred, (-1,)).tolist())
        y_true_total.extend(np.reshape(y_true, (-1,)).tolist())

    return accuracy_score(y_true_total, y_pred_total)


class Evaluator(Callback):
    """
    继承底层的callback类，构建模型选择器
    """

    def __init__(self, ):
        Callback.__init__(self, )
        self.best = 0.
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """
        在每个批次开始，判断批次步数，来调整训练时的学习率，第一个循环warmup,第二个循环降到最小
        :param batch: 批次步数
        :param logs: 日志信息
        :return: None
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * match_train_config.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (match_train_config.learning_rate - match_train_config.min_learning_rate)
            lr += match_train_config.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        每个循环结束时进行处理，选择accuracy指标最大的模型进行保存
        :param epoch: 循环数
        :param logs: 日志
        :return: None
        """

        dev_accuracy = evaluate(DEV_DATAGENERATOR)
        # 如果验证集准确率比历史最高还要高，则保存模型
        if dev_accuracy > self.best:
            self.best = dev_accuracy
            MATCH_MODEL.save(TRAINED_MODEL_PATH, include_optimizer=True)
        test_accuracy = evaluate(TEST_DATAGENERATOR)
        logger.info(
            f'dev_accuracy:{dev_accuracy}.4f, best_dev_accuracy:{self.best}.4f, test_f_score: {test_accuracy}.4f\n')

    @staticmethod
    def flat_lists(lists):
        """
         对传入的列表进行拉平。
        :param lists: (list)传入二维列表
        :return: all_elements（list)拉平后的列表
        """

        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements


def model_train():
    """
    模型训练
    :return:None
    """
    # 构建评估对象
    evaluator = Evaluator()
    # 判断模型是否是第一次训练，是否需要重载,如果不需要重载则做第一次训练，否则，重载模型做二次训练
    if not os.path.exists(match_train_config.first_trained_model_path):
        logger.info("开始首次训练模型！")
    else:
        logger.info("开始加载首次训练的模型。。。")
        MATCH_MODEL.load_weights(match_train_config.first_trained_model_path)
        logger.info("开始二次训练模型！")

    # 开始模型训练
    MATCH_MODEL.fit_generator(TRAIN_DATAGENERATOR.forfit(),
                              steps_per_epoch=TRAIN_DATAGENERATOR.__len__(),
                              epochs=match_train_config.epoch,
                              callbacks=[evaluator])
    # 使用测试集评估模型效果
    accuracy = evaluate(TEST_DATAGENERATOR)
    logger.info(f'accuracy_score: {accuracy}.4f\n')


if __name__ == '__main__':
    model_train()
