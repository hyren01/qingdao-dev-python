# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np


def __predict_sample(encoder, decoder, input_sample, n_classes, output_len):
    """
    单个样本预测
    Args:
      encoder: encoder 模型
      decoder: decoder 模型
      input_sample: encoder 模型的输入样本,
      n_classes: 事件类别个数
      output_len: decoder 输出长度, 即预测天数

    Returns:
      单个样本的预测结果, shape(预测天数, 事件类别数)
    """
    state = encoder.predict(np.array([input_sample]))
    target_seq = np.array([0.0 for _ in range(n_classes)]).reshape(
        [1, 1, n_classes])
    output = []
    for _ in range(output_len):
        yhat, h, c = decoder.predict([target_seq] + state)
        # 因为模型输出固定的，所以下面代码取值是固定的
        output.append(yhat[0, 0, :])
        state = [h, c]
        target_seq = yhat
    return output


def predict_samples(encoder, decoder, inputs, output_len, n_classes):
    """
    提供预测服务
    Args:
      encoder: encoder 模型
      decoder: decoder 模型
      inputs: 预测输入数据, 为降维后的数据表数据, shape(样本数, 降维维度)
      n_classes: 事件类别数
      output_len: decoder 输出长度, 即预测天数

    Returns:
        每个事件每一天发生的概率
        输入样本的预测结果, shape(预测样本数, 预测天数, 数据中所有事件类别个数)
    """
    preds = [__predict_sample(encoder, decoder, inputs_sample, n_classes, output_len) for inputs_sample in inputs]
    return np.array(preds)
