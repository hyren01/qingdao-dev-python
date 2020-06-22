#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件cameo模型训练代码
"""
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Dense, Dropout
from keras.models import Model
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from bert4keras.models import build_transformer_model
from bert4keras.backend import K
from bert4keras.optimizers import Adam
from feedwork.utils import logger
from jdqd.common.event_emm.model_utils import TOKENIZER, generate_trained_model_path
from jdqd.a04.event_extract.algor.train.utils.event_cameo_data_util import DataGenerator, get_data
import jdqd.a04.event_extract.config.CameoTrainConfig as cameo_train_config
import jdqd.common.event_emm.BertConfig as bert_config

# 构建默认图和会话
GRAPH = tf.Graph()
SESS = tf.Session(graph=GRAPH)

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = generate_trained_model_path(cameo_train_config.trained_model_dir, cameo_train_config.trained_model_name)
# 加载训练集和验证集
TRAIN_DATA, DEV_DATA, ID2LABEL, LABEL2ID = get_data(cameo_train_config.train_data_path, cameo_train_config.dev_data_path, cameo_train_config.label2id_path,
                                                    cameo_train_config.id2label_path)


def build_model():
    """
    构建模型主体。
    :return: 模型对象
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 搭建bert模型主体
            bert_model = build_transformer_model(
                config_path=bert_config.config_path,
                checkpoint_path=bert_config.checkpoint_path,
                return_keras_model=False,
                model=bert_config.model_type
            )

            # l为模型内部的层名，格式为--str
            for l in bert_model.layers:
                bert_model.model.get_layer(l).trainable = True
            # 取出[CLS]对应的向量用来做分类
            t = Lambda(lambda x: x[:, 0])(bert_model.model.output)
            t = Dropout(cameo_train_config.drop_out_rate)(t)
            # 预测事件cameo
            cameo_out_put = Dense(len(ID2LABEL), activation='softmax')(t)
            # cameo模型主体
            cameo_model = Model(bert_model.model.inputs, cameo_out_put)

            cameo_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=Adam(cameo_train_config.learning_rate),
                                metrics=['accuracy'])

            cameo_model.summary()

    return cameo_model


# 构建模型
MODEL = build_model()


def measure():
    """
    调用模型，预测验证集结果。
    :return: result（list）验证集预测结果
    """
    result = []
    # 遍历验证数据集
    for once in DEV_DATA:
        text = once[0]
        # 事件短句进行编码
        text_token_ids, text_segment_ids = TOKENIZER.encode(first_text=text[:cameo_train_config.maxlen])
        # 调用模型预测事件cameo，[[0.001,0.005, ]]
        pre = MODEL.predict([np.array([text_token_ids]), np.array([text_segment_ids])])
        # 获取事件cameo_id(int)
        pre = np.argmax(pre)
        # 将预测结果填充到result列表中
        result.append(pre)

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
            lr = (self.passed + 1.) / self.params['steps'] * cameo_train_config.learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        # 当训练或的步数大于一个循环设定的步数且小于两倍设定步数时，也就是第二个循环学习率随步数线性减小至最小学习率
        # 在接下来的训练中一直保持最小学习率
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (cameo_train_config.learning_rate - cameo_train_config.min_learning_rate)
            lr += cameo_train_config.min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        每个循环结束时判断指标是否最好，是则将模型保存下来。
        :param epoch: 循环次数
        :param logs: 日志信息
        :return: None
        """
        # 计算验证集准确率
        accuracy = self.evaluate()
        # 如果验证集准确率比历史最高还要高则保存模型
        if accuracy > self.best:
            self.best = accuracy
            MODEL.save(TRAINED_MODEL_PATH, include_optimizer=True)
        logger.info(f'accuracy: {accuracy}.4f, best accuracy: {self.best}.4f\n')

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

    @staticmethod
    def evaluate():
        """
        评估模型在验证集上的准确率。
        :return: accuracy(float)验证集准确率
        """
        # 验证集预测值
        y_pred = measure()
        # 验证集真实标签值
        y_true = [int(once[1]) for once in DEV_DATA]
        # 验证集准确率
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy


def model_train():
    """
    进行模型训练。
    :return: None
    """
    with SESS.as_default():
        with SESS.graph.as_default():
            # 构造训练集数据生成器
            train_d = DataGenerator(TOKENIZER, cameo_train_config.maxlen, TRAIN_DATA, cameo_train_config.batch_size)
            evaluator = Evaluate()
            # 模型训练
            MODEL.fit_generator(train_d.__iter__(), steps_per_epoch=train_d.__len__(), epochs=40, callbacks=[evaluator])
            # 模型重载
            MODEL.load_weights(TRAINED_MODEL_PATH)
            # 最后调用模型预测测试集，输出最优模型在验证集上的准确率
            accuracy = evaluator.evaluate()
            logger.info(f"accuracy:{accuracy}")


if __name__ == '__main__':
    model_train()
