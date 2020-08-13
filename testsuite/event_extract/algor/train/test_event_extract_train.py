#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年08月10
import os
from jdqd.a04.event_extract.algor.train.event_extract_train import build_model, Evaluate, SESS, model_train
from jdqd.a04.event_extract.config import ExtractTrainConfig as extract_train_config
from jdqd.a04.event_extract.config import PredictConfig as extract_pred_config
from jdqd.a04.event_extract.algor.train.utils.event_extract_data_util import get_data
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path


def test_extract_model_train():
    """
    调用事件抽取模块训练函数，测试训练流程是否成功
    :return: status
    """
    # 训练后模型路径
    trained_model_path = cat_path(extract_train_config.trained_model_dir, f"extract_model_{0}_{0}")
    # 模型训练
    model_train(version="0", model_id="0", all_steps=10, trained_model_dir=extract_train_config.trained_model_dir,
                data_dir=extract_train_config.supplement_data_dir, maxlen=extract_train_config.maxlen,
                epoch=1, batch_size=extract_train_config.batch_size,
                max_learning_rate=extract_train_config.learning_rate,
                min_learning_rate=extract_train_config.min_learning_rate, model_type="roberta")
    # 判断是否有模型生成
    assert os.path.exists(trained_model_path) == True
    # 将生成的模型删除
    os.remove(trained_model_path)
    logger.info(f"event_extract_train demo is OK!")

    return {"status": "success"}


def test_model(version, model_id, trained_model_dir="", data_dir=extract_train_config.supplement_data_dir, maxlen=160):
    """
    进行模型训练的主函数，搭建模型，加载模型数据，根据传入的参数进行模型测试，测试模型是否达标
    :param data_dir: 补充数据的文件夹路径
    :param trained_model_dir: 训练后模型存放路径
    :param maxlen: 最大长度
    :param version: 模型版本号
    :param model_id: 模型id
    :return: status, F1, precision, recall, corpus_num
    """
    # 训练后模型路径
    if version and model_id:
        trained_model_path = cat_path(trained_model_dir, f"extract_model_{version}_{model_id}")
    else:
        trained_model_path = extract_pred_config.event_extract_model_path

    # 获取训练集、验证集
    train_data, dev_data = get_data(extract_train_config.train_data_path, extract_train_config.dev_data_path,
                                    data_dir)
    # 搭建模型
    trigger_model, object_model, subject_model, loc_model, time_model, negative_model, train_model = build_model()

    with SESS.as_default():
        with SESS.graph.as_default():
            # 构造callback模块的评估类
            evaluator = Evaluate(dev_data, maxlen, trained_model_path,
                                 trigger_model, object_model, subject_model, loc_model, time_model, negative_model,
                                 train_model)

            # 重载模型参数
            train_model.load_weights(trained_model_path)
            # 将验证集预测结果保存到文件中，暂时注释掉
            f1, precision, recall = evaluator.evaluate()

            assert f1 >= 0.8
            assert precision >= 0.8
            assert recall >= 0.8

            logger.info(f"f1:{f1}, precision:{precision}, recall:{recall}")
            logger.info(f"model is OK!")

    return {"status": "success", "version": version, "model_id": model_id,
            "results": {"f1": f1, "precison": precision, "recall": recall}}


if __name__ == "__main__":
    test_extract_model_train()
    test_model(version='', model_id='', trained_model_dir="", data_dir=extract_train_config.supplement_data_dir,
               maxlen=160)
