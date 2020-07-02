# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import os
import feedwork.AppinfoConf as appconf
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
from datetime import timedelta, datetime
from jdqd.a03.event_pred.algor.train import train_model
from jdqd.a03.event_pred.algor.common import preprocess
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, preprocess as pp
from jdqd.a03.event_pred.algor.train.model_evalution import evaluate_sub_models as meval
from jdqd.a03.event_pred.bean.t_event_model import EventModel
from jdqd.a03.event_pred.algor.common.model_util import load_models
from jdqd.a03.event_pred.algor.predict.predict import predict_sample


# 模型存放路径
models_dir = cat_path(appconf.ALGOR_MODULE_ROOT, 'event_pred')
date_formatter = '%Y-%m-%d'


def train_over_hyperparameters(data, dates, events_set, events_p_oh, event_model: EventModel):
    """
    执行页面的训练模型请求, 遍历不同超参数的组合来训练模型. 先对降维维度遍历,
    再对encoder输入长度遍历, 产生的子模型数量为降维维度遍历个数与encoder输入长度遍历个数之积.
    随encoder输入长度, 降维维度的增加, 模型训练时间会变长. epoch越大, 模型训练时间越长
    Args:
      data: 模型目录
      dates: 模型目录
      events_set: 模型目录
      events_p_oh: 模型目录
      event_model: EventModel实体类

    Returns:
      训练完成的模型文件所在目录列表
      模型名称列表
      每个模型对应的 decoder 输出序列列表. 不同的模型由于输入序列长度不同导致输出序列不同
      每个模型对应的超参数列表
    """
    logger.info('开始训练模型')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    pca_dir = models_dir
    sub_model_dirs = []
    sub_model_names = []
    lag_dates = []
    pcas = []
    outputs_list = []
    params_list = []
    # 先进行降维，对于PCA降维来说它始终只有固定的降维组合，若先对滞后期的选择的话，会导致出现N个重复的降维组合
    for i in range(event_model.dr_min, event_model.dr_max, event_model.size):  # 各种降维选择
        values_pca = preprocess.apply_pca(i, pca_dir, data)
        # 基于页面选择的开始日期、结束日期的整个范围中每一天作为一个基准日期，在该基准日期往前推max_input_len至min_input_len天
        # 的范围内每次间隔5天（10、15、20天）拉取数据训练模型。
        for j in range(event_model.delay_min_day, event_model.delay_max_day, 5):  # 滞后期的选择
            logger.info(f"Current value: 滞后期={j}, pca={i}")
            lag_dates.append(j)
            pcas.append(i)
            sub_model_name = f'{event_model.model_name}-{j}-{event_model.days}-{i}'
            sub_model_names.append(sub_model_name)
            sub_model_dir = cat_path(models_dir, sub_model_name)
            if not os.path.exists(sub_model_dir):
                os.makedirs(sub_model_dir)
            sub_model_dirs.append(sub_model_dir)
            array_x, array_y, array_yin = train_model.gen_samples(values_pca, events_p_oh, j, event_model.days,
                                                                  dates, event_model.tran_start_date,
                                                                  event_model.tran_end_date)
            outputs_list.append(array_y)
            params_list.append([j, event_model.days, i])
            train_model.train(event_model.train_batch_no, event_model.epoch, event_model.neure_num, array_x,
                              array_y, array_yin, sub_model_dir)

    logger.info('训练完成, 模型存入数据库')

    detail_ids = pgsql.model_train_done(event_model.model_id, lag_dates, pcas, sub_model_dirs, sub_model_names,
                                        outputs_list, events_set)

    return sub_model_dirs, params_list, detail_ids


def evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set, eval_start_date,
                        eval_end_date):
    """
    用指定的模型对指定的数据进行评估。

    :param data: array.特征数据
    :param dates: datetime/date.数据表日期列表
    :param detail_ids: array.t_event_model_detail表的id.
    :param sub_model_dirs: array.模型存放目录
    :param params_list: array.超参数集合
    :param events_p_oh: array, one-hot 形式的补全事件列表
    :param events_set: array.去重且排序后的事件类别列表
    :param eval_start_date: string.评估的数据开始日期
    :param eval_end_date: string.评估的数据结束日期
    """
    logger.info('开始评估模型')
    n_classes = len(events_set)

    # 评估模型. scores: 子模型综合评分列表; events_num: 测试机事件个数
    scores, events_num = meval(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set,
                               n_classes, eval_start_date, eval_end_date)

    logger.info('模型评估结束, 筛选top模型并入库')
    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:min(10, len(scores))]

    pgsql.model_eval_done(top_scores, events_num)


def web_predict(model_id, data, dates, events_set, task_id, pred_start_date):
    """
    # TODO 这个注释跟没写一样，下面的参数没更新
    通过页面传入的参数使用指定模型进行预测, 并将预测结果存入数据库.
    Args:
      model_id: 预测使用的模型 id
      data: 预测使用的模型 id
      dates: 预测使用的模型 id
      events_set: 预测使用的模型 id
      task_id: 预测任务的 id
      pred_start_date: 开始预测日期, 即预测结果由此日期开始
    """
    event_predict_array = pgsql.query_sub_models_by_model_id(model_id)
    # 事件类别数量(含0事件)
    num_classes = len(events_set)

    preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred, pred_detail_ids, last_date_data_pred = \
        __predict_by_sub_models(data, dates, event_predict_array, pred_start_date, num_classes, task_id)
    if pred_detail_ids:
        pgsql.insert_pred_result(preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred,
                                 pred_detail_ids, events_set, task_id)
    return last_date_data_pred


def __predict_by_sub_models(data, dates, event_predict_array: list, pred_start_date, num_classes, task_id):
    """
    使用各个子模型进行预测。
    Args:
      data: 预测输入数据
      dates: 数据表日期列表
      event_predict_array: array. EventPredict实体类，封装预测时需要用到的信息
      pred_start_date: 开始预测日期, 即此日期后均有预测结果
      num_classes: 事件类别数量
      task_id: 由页面传入

    Returns:
      preds_one: 所有子模型预测结果, 每天为输入数据预测的后一天结果, 如果最后一个样本没有历史预测结果,
          则最后一个样本的预测结果全部保留.
          shape(最后样本有历史预测结果): (子模型数, 未预测的样本数),
          或shape(最后样本无历史预测结果): (子模型数, 未预测的样本数 + output_len - 1)
      preds_all: 所有子模型预测结果, 每天为输入数据预测的多天结果. shape: (子模型数, 未预测的样本数, 预测天数)
      dates_pred_one: preds_one 对应的预测日期, shape 与 preds_one 相同
      dates_pred_all: preds_all 对应的预测日期, shape 与 preds_all 相同
      dates_pred_data: preds_one 对应的数据结束日期, shape 与 preds_one 相同
      pred_detail_ids: 预测的子模型 detail_id 列表, 去掉了已预测的子模型
      last_date_data_pred: 预测所用数据最后一天日期
    """
    preds_one = []
    preds_all = []
    dates_pred_one = []
    dates_pred_all = []
    dates_pred_data = []
    pred_detail_ids = []
    last_date_data_pred = None

    for event_predict in event_predict_array:
        sub_model = event_predict.model_name
        logger.info(f'正在使用模型{sub_model}进行预测')
        input_len = event_predict.lag_date
        # 预测天数指的是模型可以预测n天，而不是预测开始日期+n天。假设预测开始日期为6月1日，则模型从6月1日起，每次预测n天
        # 直到今天的日期，且对重复预测的处理是使用最新的预测。
        output_len = event_predict.days
        n_pca = event_predict.pca
        detail_id = event_predict.detail_id

        model_dir = cat_path(models_dir, sub_model)
        values_pca = pp.apply_pca(n_pca, models_dir, data, True)
        inputs_data, output_dates = pp.gen_inputs_by_pred_start_date(values_pca, input_len, dates, pred_start_date)
        # 取样本数据中最大的日期，再往后推1天  TODO dates[-1]要求日期必须升序排序
        max_output_date = datetime.strptime(dates[-1], date_formatter).date() + timedelta(1)
        output_dates.append(max_output_date)  # 此时output_dates不包含预测第一天后日期
        dates_data = [datetime.strptime(output_dates[0], date_formatter).date() - timedelta(1)]
        dates_data.extend([datetime.strptime(out_put_date, date_formatter).date()
                           for out_put_date in output_dates[:-1]])

        last_date_data_pred = dates_data[-1]

        predicted_detail_id_dates = pgsql.query_predicted_rsts(detail_id, pred_start_date, task_id)
        predicted_dates = predicted_detail_id_dates.get(detail_id)  # type of list of str
        if predicted_dates is None:
            latest_date_predicted = False
            predicted_dates_to_delete = []
        else:
            predicted_dates = sorted([pp.parse_date_str(d) for d in predicted_dates])
            predicted_dates_to_delete = predicted_dates[-output_len + 1:]
            predicted_dates = predicted_dates[:-output_len + 1]  # 截取只预测一天的预测结果
            max_predicted_date = predicted_dates[-1]
            zipped_unpredicted = [[d, i, dd] for d, i, dd in zip(output_dates, inputs_data, dates_data)
                                  if d not in predicted_dates]
            if not zipped_unpredicted:
                logger.info(f'{sub_model}所有日期已预测, 跳过')
                continue

            output_dates, inputs_data, dates_data, = zip(*zipped_unpredicted)
            output_dates = list(output_dates)
            inputs_data = list(inputs_data)
            dates_data = list(dates_data)
            if max_predicted_date == max_output_date:
                latest_date_predicted = True
            else:
                latest_date_predicted = False

        # 预测日期, 包含第一天以后日期
        dates_pred_all_model = [[(dd + timedelta(t)) for t in range(1, output_len + 1)] for dd in dates_data]

        encoder, decoder = load_models(model_dir)
        pred = model_predict(encoder, decoder, inputs_data, output_len, num_classes)

        pred_one = [p[0] for p in pred]  # 在预测到最后一天之前的每一天预测的结果都只有第一天可用
        if not latest_date_predicted:
            pred_one.extend(pred[-1][1:])
            # 此时output_dates添加第一天以后日期
            output_dates.extend([max_output_date + timedelta(d) for d in range(1, output_len)])
            dates_data.extend([dates_data[-1]] * (output_len - 1))
            if predicted_dates_to_delete:
                pgsql.delete_predicted_dates(detail_id, predicted_dates_to_delete)

        pred_detail_ids.append(detail_id)
        preds_one.append(pred_one)
        preds_all.append(pred)
        dates_pred_one.append(output_dates)
        dates_pred_data.append(dates_data)
        dates_pred_all.append(dates_pred_all_model)

    return preds_one, preds_all, dates_pred_one, dates_pred_all, dates_pred_data, pred_detail_ids, last_date_data_pred


def model_predict(encoder, decoder, inputs, output_len, n_classes):
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
        输入样本的预测结果, shape(一次预测结果, 预测天数, 数据中所有事件类别个数)
        一次预测结果包含：预测5天，每天9个事件类别
    """
    preds = [predict_sample(encoder, decoder, inputs_sample, n_classes, output_len) for inputs_sample in inputs]
    return preds
