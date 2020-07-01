#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import flask
import traceback
import feedwork.AppinfoConf as appconf

from flask import request
from feedwork.utils import logger
from keras import backend as K
from jdqd.a03.event_pred.services.model_helper import train_over_hyperparameters, evaluate_sub_models, web_predict
from jdqd.a03.event_pred.services.data_helper import combine_data, get_event
from jdqd.a03.event_pred.algor.common.pgsql_util import query_teventmodel_by_id, query_teventtask_by_id


webApp = flask.Flask(__name__)


@webApp.route("/buildModel", methods=['GET', 'POST'])
def build_model():
    """
    训练模型接口。该接口完成了三个主要步骤：构建数据集、训练模型、评估模型。
    1、构建数据集。获取数据库中指定的训练及评估数据，并且对数据集进行转换；
    2、训练模型。根据PCA降维的特征选择以及max_input_len、min_input_len进行组合，每个组合都会训练出一个模型并且记录到数据库；
    3、评估模型。对每个训练出的模型进行评估，评估结果记录到数据库。
    返回数据，如：{"status":"success", "model_path":[]}
    """

    model_id = request.form.get("model_id")
    logger.info(f"开始根据模型id<{model_id}>构建模型")
    try:
        # 查询t_event_model表，获取模型详细信息，该方法返回的是EventModel实体
        event_model = query_teventmodel_by_id(model_id)
        # 1、构建数据集。获取数据库中指定的训练及评估数据，并且对数据集进行转换；
        date, data = combine_data(event_model.tables_name)
        events_set, events_p_oh = get_event(date, event_model.event_type)
        # 2、训练模型。根据PCA降维的特征选择以及max_input_len、min_input_len进行组合，每个组合都会训练出一个模型并且记录到数据库；
        sub_model_dirs, params_list, detail_ids = train_over_hyperparameters(data, date, events_set, events_p_oh,
                                                                             event_model)
        # 3、评估模型。对每个训练出的模型进行评估，评估结果记录到数据库。
        evaluate_sub_models(model_id, data, date, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set,
                            event_model.evaluation_start_date, event_model.evaluation_end_date)

        logger.info(f"当前模型id<{model_id}>的模型构建完成")
        return {"success": True, "model_path": sub_model_dirs}
    except Exception as e:
        logger.error(f"表id {model_id} 训练发生异常：{traceback.format_exc()}")
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


@webApp.route("/modelPredict", methods=['GET', 'POST'])
def model_predict():
    """
    模型预测接口。该接口完成了三个主要步骤：构建数据集、模型预测。
    1、构建数据集。获取数据库中指定的预测数据，并且对数据集进行转换；
    2、模型预测。使用指定的模型及数据集进行预测，预测结果记录到数据库。
    返回数据，如：{"status":"success"}
    """
    task_id = request.form.get("task_id")

    logger.info(f"开始根据模型预测id<{task_id}>数据进行预测")
    try:
        # 查询t_event_task表，获取模型任务信息，该方法返回的是EventTask实体
        event_task = query_teventtask_by_id(task_id)
        # 1、构建数据集。获取数据库中指定的预测数据，并且对数据集进行转换；
        dates, data = combine_data(event_task.tables_name)
        events_set, events_p_oh = get_event(dates, event_task.event_type)
        # 2、模型预测。使用指定的模型及数据集进行预测，预测结果记录到数据库。
        web_predict(event_task.model_id, data, dates, events_set, task_id, event_task.sample_start_date)
        logger.info(f"模型id<{event_task.model_id}>的预测完成")
        return {"success": True}
    except Exception as e:
        logger.error(f"表 id {task_id} 预测发生异常：{traceback.format_exc()}")
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


if __name__ == '__main__':
    server_addr = appconf.appinfo["a03"]['server_addr']
    webApp.config['JSON_AS_ASCII'] = False
    webApp.run(host=server_addr['host'], port=server_addr['port'])

