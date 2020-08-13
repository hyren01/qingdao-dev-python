# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:36:09 2020

@author: 12894
"""

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D
from keras import regularizers
from keras.layers.merge import concatenate
from feedwork.utils.FileHelper import cat_path
from keras.callbacks import Callback
from jdqd.a03.event_pred.algor.train import model_evalution
from jdqd.a03.event_pred.algor.common import preprocess
import numpy as np
from feedwork.utils import logger
from jdqd.a03.event_pred.algor.predict import predict_cnn


def __define_model(Input_shape1, Input_shape2, m, k):
    """
    构建模型网络主体
    :param Input_shape1: 样本数据步长
    :param Input_shape2: 样本数据每步长维度
    :param m: 卷积核大小
    :param k: 过滤器大小
    :return 模型主体
    """
    model_inputs = Input(shape=(Input_shape1, Input_shape2))
    cnn1 = Conv1D(225, m, padding='same', strides=1, activation='relu',
                  kernel_regularizer=regularizers.l1(0.00001))(
        model_inputs)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=k)(cnn1)
    cnn2 = Conv1D(225, m, padding='same', strides=1, activation='relu',
                  kernel_regularizer=regularizers.l1(0.00001))(
        model_inputs)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=k)(cnn2)
    cnn3 = Conv1D(225, m, padding='same', strides=1, activation='relu',
                  kernel_regularizer=regularizers.l1(0.00001))(
        model_inputs)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = MaxPool1D(pool_size=k)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    dropout = Dropout(0.5)(cnn)
    flatten = Flatten()(dropout)
    dense = Dense(128, activation='relu')(flatten)
    dense = BatchNormalization()(dense)
    dropout = Dropout(0.5)(dense)
    tensor_output = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=model_inputs, outputs=tensor_output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train(train_input, train_output, kernel_size, pool_size, model_dir,
          batch_size, epochs, data, events_p_oh, input_len, output_len, dates,
          eval_start_date, eval_end_date, eval_event, events_set):
    """
    循环遍历参数训练模型，选择最优模型保存
    :param events_set: 全量数据的事件类别
    :param train_input: 训练数据输入
    :param train_output: 训练数据输出
    :param kernel_size: 卷积核尺寸
    :param pool_size: 池化尺寸
    :param batch_size: 训练批次大小
    :param epochs: 训练次数
    :param model_dir: 模型保存目录
    """

    Input_shape1 = train_input.shape[1]
    Input_shape2 = train_input.shape[2]

    # 训练模型
    model = __define_model(Input_shape1, Input_shape2, kernel_size, pool_size)
    callback = Evaluate(model_dir, data, events_p_oh, input_len,
                 output_len, dates, eval_start_date, eval_end_date, eval_event,
                        events_set)
    model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs,
              callbacks=[callback], verbose=2)


class Evaluate(Callback):
    """
    继承Callback类，改下内部方法，使得当随着训练步数增加时，选择并保存最优模型
    """

    def __init__(self, model_dir, data, events_p_oh, input_len,
                 output_len, dates, eval_start_date, eval_end_date, eval_event,
                 events_set):
        self.best = -1.
        self.model_dir = model_dir
        self.data = data
        self.events_p_oh = events_p_oh
        self.input_len = input_len
        self.output_len = output_len
        self.dates = dates
        self.eval_start_date = eval_start_date
        self.eval_end_date = eval_end_date
        self.eval_event = eval_event
        self.events_set = events_set

    def on_epoch_end(self, epoch, logs=None):
        """
        选择所有训练次数中，f1最大时的模型
        :param epoch: 训练次数

        return None
        """
        score_summary = self.evaluate()
        if score_summary > self.best:
            logger.info(f'{score_summary} better than old: {self.best}')
            self.best = score_summary
            model_path = cat_path(self.model_dir, 'cnn_model.h5')
            self.model.save(model_path, include_optimizer=True)

    def evaluate(self):
        inputs_test, outputs_test = \
            preprocess.gen_samples_by_pred_date(self.data,
                                                self.events_p_oh,
                                                self.input_len,
                                                self.output_len,
                                                self.dates,
                                                self.eval_start_date,
                                                self.eval_end_date)
        event_col = self.events_set.index(self.eval_event)
        outputs_test = outputs_test[:, :, event_col]
        outputs_test = np.reshape(outputs_test, [*outputs_test.shape, 1])
        events_num = preprocess.get_event_num(outputs_test, [self.eval_event])
        preds = predict_cnn.predict_samples(self.model, inputs_test)
        evals_summary, evals_separate, eval_events = \
            model_evalution.evaluate_sub_model(
                preds,
                outputs_test,
                [self.eval_event],
                events_num)
        return evals_summary[0]
