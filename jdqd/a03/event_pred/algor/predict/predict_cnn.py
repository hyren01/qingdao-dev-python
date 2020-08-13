# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:41:28 2020

@author: 12894
"""
import numpy as np


def predict_samples(model, input_data):
    """
    预测并输出测试结果
    :param model:模型对象
    :param input_data:最优参数测试输入数据
    :return 预测结果, shape(样本数, 预测天数(1天), 类别数量(1个类别))
    """
    prob = model.predict(input_data)
    shape = prob.shape
    prob = np.reshape(prob, [*shape, 1])
    return prob
