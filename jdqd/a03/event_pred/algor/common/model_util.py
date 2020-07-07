# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from keras.models import load_model
from feedwork.utils.FileHelper import cat_path


def load_models(model_dir):
    """
    加载模型文件目录下的模型
    Args:
      model_dir: 模型文件所在目录

    Returns:
      模型的 encoder, decoder
    """
    encoder = load_model(cat_path(model_dir, 'encoder.h5'), compile=False)
    decoder = load_model(cat_path(model_dir, 'decoder.h5'), compile=False)
    return encoder, decoder
