#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import flask
import numpy as np
import traceback
import feedwork.AppinfoConf as appconf

from flask import request
from feedwork.utils import logger
from keras import backend as K
from jdqd.a03.event_pred.algor.common import obtain_data as od
from jdqd.a03.event_pred.services.model_helper import \
    transform_data, train_over_hyperparameters, evaluate_sub_models, web_predict

webApp = flask.Flask(__name__)


@webApp.route("/buildModel", methods=['GET', 'POST'])
def build_model():
    """
    接收 http 请求, 根据请求内容遍历各种参数训练模型
    TODO 该接口有两个重要步骤：1、训练模型；2、评估模型。这里应该明确写出该信息以及其重要事项，比如会更新什么表、有没有数据库事务之类的
        PS:个人觉得该接口的两个重要步骤要在下面的代码块中详细写清楚，方法注释中稍微提一下即可。
        该接口的构建训练数据集、测试数据集存在特殊情况且应该写明，总数据集3年的数据才1000多条，所以不需要单独建立数据子集。
    Returns:
      任务状态信息
    """
    # TODO 缺关键参数的判断
    model_id = request.form.get("model_id")
    model_name = request.form.get("model_name")
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    # TODO 以下很多参数通过model_id查询出来，不需要传值
    # event_type = request.form.get("event_type")
    output_len = request.form.get("days")
    output_len = int(output_len)
    min_dim = request.form.get("min_dim")  # 最小降维
    min_dim = 5 if min_dim is None else int(min_dim)
    max_dim = request.form.get("max_dim")  # 最大降维
    max_dim = 10 if max_dim is None else int(max_dim)
    min_input_len = request.form.get("min_lag")  # 最小滞后期
    min_input_len = int(min_input_len) if min_input_len else 10
    max_input_len = request.form.get("max_lag")  # 最大滞后期
    max_input_len = int(max_input_len) if max_input_len else 61
    num_units = request.form.get("unit")  # 神经元个数
    num_units = int(num_units) if num_units else 128
    batch = request.form.get("batch")  # 批量数据个数
    batch = int(batch) if batch else 64
    epoch = request.form.get("epoch")  # 训练次数
    epoch = int(epoch) if epoch else 150
    step = request.form.get('size')     # 降维维度遍历步长
    step = int(step) if step else 4

    # TODO 缺参数间的逻辑判断，比如开始日期不能大于结束日期
    train_start_date = request.form.get('tran_start_date')  # 训练数据开始预测日期
    train_end_date = request.form.get('tran_end_date')  # 训练数据结束预测日期
    eval_start_date = request.form.get('evaluation_start_date')     # 评估数据开始预测日期
    eval_end_date = request.form.get('evaluation_end_date')     # 评估数据结束预测日期
    try:
        # TODO ！！！！！！！！！！！！！！！！这里有暗坑：1、下面这行代码会判断每一列是否全为空列来决定该列是否参加模型训练，
        #  这个事情在业务角度应该能确认哪些列一定为空所以在数据库表层面删除；2、下面这行代码会拉取数据表中所有数据，然后在程序
        #  中进行简单条件筛选，这是错的！
        dates, data = od.combine_data(tables)
        events_set, events_p_oh = transform_data(dates, model_id)

        sub_model_dirs, params_list, detail_ids = \
            train_over_hyperparameters(model_id, data, dates, events_set, events_p_oh, model_name,
                                       train_start_date, train_end_date, output_len, min_dim, max_dim, min_input_len,
                                       max_input_len, step, num_units, batch, epoch)

        evaluate_sub_models(model_id, data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set,
                            eval_start_date, eval_end_date)

        logger.info(f"当前表 {','.join(tables)} 的模型构建完成")
        return {"success": True, "model_path": sub_model_dirs}
    except Exception as e:
        logger.error(f"表 {','.join(tables)} 训练发生异常：{traceback.format_exc()}")
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


@webApp.route("/modelPredict", methods=['GET', 'POST'])
def model_predict():
    """
    TODO same
    接收页面的预测 http 请求
    Returns:
      任务状态信息
    """
    # TODO same
    model_id = request.form.get("model_id")
    model_id = str(model_id)
    tables = request.form.get("tables")
    tables = np.array(str(tables).split(","))
    task_id = request.form.get("task_id")
    pred_start_date = request.form.get("sample_start_date")

    logger.info(f"开始根据表 {','.join(tables)} 数据进行预测")
    try:
        dates, data = od.combine_data(tables)
        events_set, events_p_oh = transform_data(dates, model_id)

        web_predict(model_id, data, dates, events_set, tables, task_id, pred_start_date)
        return {"success": True}
    except Exception as e:
        logger.error(f"表 {','.join(tables)} 预测发生异常：{traceback.format_exc()}")
        return {"success": False, "exception": e}
    finally:
        K.clear_session()


if __name__ == '__main__':
    server_addr = appconf.appinfo["a03"]['server_addr']
    webApp.config['JSON_AS_ASCII'] = False
    webApp.run(host=server_addr['host'], port=server_addr['port'])

