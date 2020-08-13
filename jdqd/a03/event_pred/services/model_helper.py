# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import os
import numpy as np
import feedwork.AppinfoConf as appconf
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
from feedwork.utils.UuidHelper import guid
from datetime import timedelta, datetime
from jdqd.a03.event_pred.algor.train import train_model_rnn, train_model_cnn
from jdqd.a03.event_pred.algor.common import preprocess
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, \
    preprocess as pp
from jdqd.a03.event_pred.algor.train.model_evalution import evaluate_models
from jdqd.a03.event_pred.bean.t_event_model import EventModel
from jdqd.a03.event_pred.algor.common.model_util import load_rnn_models, \
    load_cnn_model
from jdqd.a03.event_pred.algor.predict.predict_rnn import \
    predict_samples as predict_sample_rnn
from jdqd.a03.event_pred.algor.predict.predict_cnn import \
    predict_samples as predict_sample_cnn
from jdqd.a03.event_pred.enum.model_type import ModelType

# 模型存放路径
base_model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, 'event_pred')
date_formatter = '%Y-%m-%d'

cnn_day = 1  # cnn模型只能预测一天，使用该参数标识


def __train_over_hyperparameters_RNN(data, dates, events_set, events_p_oh,
                                     event_model: EventModel):
    """
    提供训练模型服务, 遍历不同超参数的组合来训练模型. 先对降维维度遍历, 再对encoder输入长度遍历,
    产生的子模型数量为降维维度遍历个数与encoder输入长度遍历个数之积. 随encoder输入长度, 降维维度的增加,
    模型训练时间会变长. epoch越大, 模型训练时间越长
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
    logger.info('<RNN>开始训练模型')

    # 每次训练生成一个对应的目录
    new_model_dir = cat_path(base_model_dir, guid())
    if not os.path.exists(new_model_dir):
        os.makedirs(new_model_dir)
    pca_dir = new_model_dir
    sub_model_dirs = []
    sub_model_names = []
    lag_dates = []
    pcas = []
    outputs_list = []
    params_list = []
    # 先进行降维，对于PCA降维来说它始终只有固定的降维组合，若先对滞后期的选择的话，会导致出现N个重复的降维组合
    for i in range(event_model.dr_min, event_model.dr_max,
                   event_model.size):  # 各种降维选择
        values_pca = preprocess.apply_pca(i, pca_dir, data)
        # 基于页面选择的开始日期、结束日期的整个范围中每一天作为一个基准日期，在该基准日期往前推max_input_len至min_input_len天
        # 的范围内每次间隔5天（10、15、20天）拉取数据训练模型。
        for j in range(event_model.delay_min_day, event_model.delay_max_day,
                       5):  # 滞后期的选择
            logger.info(f"<RNN> Current value: 滞后期={j}, pca={i}")
            lag_dates.append(j)
            pcas.append(i)
            sub_model_name = f'{event_model.model_name}-{j}-{event_model.days}-{i}'
            sub_model_names.append(sub_model_name)
            sub_model_path = cat_path(new_model_dir, sub_model_name)
            if not os.path.exists(sub_model_path):
                os.makedirs(sub_model_path)
            sub_model_dirs.append(sub_model_path)
            # flag 表示样本是否可用
            array_x, array_y, array_yin = gen_samples(values_pca, events_p_oh, j, event_model.days, dates,
                                                      event_model.tran_start_date, event_model.tran_end_date)
            outputs_list.append(array_y)
            params_list.append([j, event_model.days, i])

            train_model_rnn.train(event_model.train_batch_no, event_model.epoch,
                                  event_model.neure_num, array_x,
                                  array_y, array_yin, sub_model_path,
                                  values_pca, events_p_oh, j, event_model.days, dates,
                                  event_model.evaluation_start_date,
                                  event_model.evaluation_end_date, events_set)

    logger.info('<RNN>训练完成, 模型存入数据库')

    detail_ids = pgsql.model_train_done_rnn(event_model.model_id, lag_dates,
                                            pcas, sub_model_dirs,
                                            sub_model_names,
                                            outputs_list, events_set,
                                            new_model_dir)

    return sub_model_dirs, params_list, detail_ids, new_model_dir


def __train_over_hyperparameters_CNN(data, dates, events_set, events_p_oh,
                                     event_model: EventModel):
    """
    提供训练模型服务, 遍历不同超参数的组合来训练模型.
    Args:
      data: ndarray.
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
    logger.info('<CNN>开始训练模型')

    # 每次训练生成一个对应的目录
    new_model_dir = cat_path(base_model_dir, guid())
    if not os.path.exists(new_model_dir):
        os.makedirs(new_model_dir)
    sub_model_dir_list = []
    sub_model_name_list = []
    lag_date_list = []
    output_list = []
    params_list = []
    kernel_size_list = []  # 卷积核
    pool_size_list = []  # 过滤器
    # 基于页面选择的开始日期、结束日期的整个范围中每一天作为一个基准日期，在该基准日期往前推max_input_len至min_input_len天
    # 的范围内每次间隔5天（10、15、20天）拉取数据训练模型。
    for n_in in range(event_model.delay_min_day, event_model.delay_max_day, 5):  # 滞后期的选择
        logger.info(f"<CNN> Current value: 滞后期={n_in}")
        # CNN只能预测一天，所以写1
        array_x, array_y, __ = gen_samples(data, events_p_oh, n_in, cnn_day, dates, event_model.tran_start_date,
                                           event_model.tran_end_date)
        if event_model.event not in events_set:
            raise RuntimeError(f"<CNN>模型训练时，event信息<{event_model.event}>必须在数据中的事件类别范围内<{','.join(events_set)}>")
        event_col = events_set.index(event_model.event)

        # 用于计算事件个数
        array_y = array_y[:, :, event_col]
        array_y_ = np.reshape(array_y, [*array_y.shape, 1])
        for k in range(event_model.pool_size_min, event_model.pool_size_max, event_model.pool_size_step):  # 过滤器的选择
            # 过滤器值不能大于滞后期
            if k > n_in:
                continue
            # 卷积核的选择
            for m in range(event_model.kernel_size_min,
                           event_model.kernel_size_max,
                           event_model.kernel_size_step):
                # 卷积核值不能大于滞后期
                if m > n_in:
                    continue
                sub_model_name = f'{event_model.model_name}-{n_in}-{cnn_day}-{k}-{m}'
                sub_model_name_list.append(sub_model_name)
                sub_model_path = cat_path(new_model_dir, sub_model_name)
                if not os.path.exists(sub_model_path):
                    os.makedirs(sub_model_path)
                sub_model_dir_list.append(sub_model_path)
                pool_size_list.append(k)
                kernel_size_list.append(m)
                lag_date_list.append(n_in)
                output_list.append(array_y_)
                params_list.append([n_in, cnn_day])
                train_model_cnn.train(array_x, array_y, m, k, sub_model_path,
                                      event_model.train_batch_no,
                                      event_model.epoch, data, events_p_oh,
                                      n_in,
                                      cnn_day, dates,
                                      event_model.evaluation_start_date,
                                      event_model.evaluation_end_date,
                                      event_model.event,
                                      events_set)

    logger.info('<CNN>训练完成, 模型存入数据库')
    events_set_ = [event_model.event]

    detail_ids = pgsql.model_train_done_cnn(event_model.model_id,
                                            kernel_size_list, pool_size_list,
                                            lag_date_list,
                                            sub_model_dir_list,
                                            sub_model_name_list, output_list,
                                            events_set_,
                                            new_model_dir)

    return sub_model_dir_list, params_list, detail_ids, new_model_dir


# ----------------------公共代码块开始-------------------------


def train_over_hyperparameters(data, date, events_set, events_p_oh,
                               event_model: EventModel):
    """
    提供训练模型服务, 遍历不同超参数的组合来训练模型.
    Args:
      data: dataframe.
      date: 模型目录
      events_set: 模型目录
      events_p_oh: 模型目录
      event_model: EventModel实体类

    Returns:
      训练完成的模型文件所在目录列表
      模型名称列表
      每个模型对应的 decoder 输出序列列表. 不同的模型由于输入序列长度不同导致输出序列不同
      每个模型对应的超参数列表
      new_model_dir: str. 模型存放位置
    """
    if ModelType.CNN.value == event_model.model_type:  # 训练CNN模型
        if event_model.event is None:
            return RuntimeError("CNN模型event信息不能为空")
        if event_model.kernel_size_max > event_model.delay_max_day or event_model.pool_size_max > \
            event_model.delay_max_day:
            return RuntimeError("CNN模型最大卷积核最大值、过滤器最大值不能大于最大滞后日期")
        # 根据过滤器、卷积核、滞后期进行组合训练，每个组合都会训练出一个模型并且记录到数据库；
        # params_list: [n_in, n_out]
        sub_model_dirs, params_list, detail_ids, new_model_dir = \
            __train_over_hyperparameters_CNN(data, date, events_set,
                                             events_p_oh, event_model)
    elif ModelType.RNN.value == event_model.model_type:  # 训练RNN模型
        # 根据PCA降维的特征选择、滞后期进行组合训练，每个组合都会训练出一个模型并且记录到数据库；
        # params_list: [n_in, n_out, n_pca]
        sub_model_dirs, params_list, detail_ids, new_model_dir = \
            __train_over_hyperparameters_RNN(data, date, events_set,
                                             events_p_oh, event_model)
    else:
        raise RuntimeError(f"Unsupport model type {event_model.model_type}")

    return sub_model_dirs, params_list, detail_ids, new_model_dir


def evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list,
                        events_p_oh, events_set,
                        eval_start_date, eval_end_date, model_type, eval_event,
                        model_dir):
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
    :param model_type: string.模型类型，对应代码项为ModelType
    :param eval_event: cnn模型专用参数
    :param model_dir: str，模型存放地址
    """
    logger.info('开始评估模型')
    n_classes = len(events_set)

    # 评估模型. scores: 子模型综合评分列表; events_num: 测试机事件个数
    scores, events_num = evaluate_models(data, dates, detail_ids,
                                         sub_model_dirs, params_list,
                                         events_p_oh, events_set,
                                         n_classes, eval_start_date,
                                         eval_end_date, model_type, eval_event,
                                         model_dir)

    if scores is None:
        raise RuntimeError('模型评估结束, 评估的事件在评估日期范围内没有发生，评估信息不能为空')
    else:
        logger.info('模型评估结束,评估信息记录入库')
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[:min(10, len(scores))]

        pgsql.model_eval_done(top_scores, events_num)


def gen_samples(data, events_p_oh, input_len, output_len, dates, pred_start_date, pred_end_date):
    """
      根据预测日期生成对应的输入与输出样本
    Args:
      data: dataframe 降维操作后的数据
      events_p_oh: 按数据表补0的事件列表, one-hot形式
      input_len: encoder 输入序列长度
      output_len: decoder 输出序列长度
      dates: 数据表对应日期列表
      pred_start_date: 开始预测日期
      pred_end_date: 预测截止日期

    Returns:
      flag表示样本是否可用
      输出序列在开始预测日期与预测截止日期之间的样本, 包含输入输出序列, 以及 decoder 训练阶段
      的 inference 输入
    """
    inputs_train, outputs_train = pp.gen_samples_by_pred_date(data, events_p_oh, input_len, output_len, dates,
                                                              pred_start_date, pred_end_date)
    # 训练阶段 inference 输入, 为样本输出序列标签延后一个时间单位, 开头以 0 填充
    # 在事件表与数据表进行join合并的时候，数据表中的某个特征可能在事件表中没有记录该特征的事件类型，
    # 意味着该特征指没有事件发生，所以填充0
    outputs_train_inf = np.insert(outputs_train, 0, 0, axis=-2)[:, :-1, :]
    return inputs_train, outputs_train, outputs_train_inf


def predict_by_model(model_id, data, dates, events_set, task_id,
                     pred_start_date, model_type, eval_event):
    """
    使用指定模型进行预测, 并将预测结果存入数据库.
    Args:
      model_id: 预测使用的模型 id
      data: 预测使用的模型 id
      dates: 预测使用的模型 id
      events_set: 预测使用的模型 id
      task_id: 预测任务的 id
      pred_start_date: 开始预测日期, 即预测结果由此日期开始
      model_type: string.模型类型，对应代码项为ModelType
      eval_event: cnn模型专用参数
    """
    event_predict_array = pgsql.query_sub_models_by_model_id(model_id)
    # CNN只预测一天
    if ModelType.CNN.value == model_type:
        events_set = [eval_event]
    # 事件类别数量(含0事件)
    num_classes = len(events_set)

    preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred, pred_detail_ids, last_date_data_pred = \
        __predict_by_sub_models(data, dates, event_predict_array,
                                pred_start_date, num_classes, task_id,
                                model_type)
    if pred_detail_ids:
        pgsql.insert_pred_result(preds, preds_all_days, dates_pred,
                                 dates_pred_all, dates_data_pred,
                                 pred_detail_ids, events_set, task_id)
    return last_date_data_pred


def __predict_by_sub_models(data, dates, event_predict_array: list,
                            pred_start_date, num_classes, task_id, model_type):
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
      model_type: string.模型类型，对应代码项为ModelType
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
        detail_id = event_predict.detail_id

        if ModelType.CNN.value == model_type:
            output_len = cnn_day
            inputs_data, output_dates = pp.gen_inputs_by_pred_start_date(data,
                                                                         input_len,
                                                                         dates,
                                                                         pred_start_date)
        elif ModelType.RNN.value == model_type:
            # 预测天数指的是模型可以预测n天，而不是预测开始日期+n天。假设预测开始日期为6月1日，则模型从6月1日起，每次预测n天
            # 直到今天的日期，且对重复预测的处理是使用最新的预测。
            output_len = event_predict.days
            values_pca = pp.apply_pca(event_predict.pca,
                                      event_predict.model_dir, data, True)
            inputs_data, output_dates = pp.gen_inputs_by_pred_start_date(
                values_pca, input_len, dates, pred_start_date)
        else:
            raise RuntimeError(f"Unsupport model type <{model_type}>")

        # 取样本数据中最大的日期，再往后推1天  TODO dates[-1]要求日期必须升序排序
        max_output_date = datetime.strptime(dates[-1],
                                            date_formatter).date() + timedelta(
            1)
        output_dates.append(max_output_date)  # 此时output_dates不包含预测第一天后日期
        dates_data = [datetime.strptime(output_dates[0],
                                        date_formatter).date() - timedelta(1)]
        dates_data.extend(
            [datetime.strptime(out_put_date, date_formatter).date()
             for out_put_date in output_dates[:-1]])
        last_date_data_pred = dates_data[-1]
        predicted_detail_id_dates = pgsql.query_predicted_rsts(detail_id,
                                                               pred_start_date,
                                                               task_id)
        predicted_dates = predicted_detail_id_dates.get(
            detail_id)  # type of list of str
        if predicted_dates is None:
            latest_date_predicted = False
            predicted_dates_to_delete = []
        else:
            predicted_dates = sorted(
                [pp.parse_date_str(d) for d in predicted_dates])
            predicted_dates_to_delete = predicted_dates[-output_len + 1:]
            predicted_dates = predicted_dates[:-output_len + 1]  # 截取只预测一天的预测结果
            max_predicted_date = predicted_dates[-1]
            zipped_unpredicted = [[d, i, dd] for d, i, dd in
                                  zip(output_dates, inputs_data, dates_data)
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
        dates_pred_all_model = [
            [(dd + timedelta(t)) for t in range(1, output_len + 1)] for dd in
            dates_data]

        sub_model_dir = cat_path(event_predict.model_dir, sub_model)
        if ModelType.RNN.value == model_type:
            encoder, decoder = load_rnn_models(sub_model_dir)
            pred = predict_sample_rnn(encoder, decoder, inputs_data, output_len,
                                      num_classes)
        elif ModelType.CNN.value == model_type:
            model = load_cnn_model(sub_model_dir)
            pred = predict_sample_cnn(model, inputs_data)

        pred_one = [p[0] for p in pred]  # 在预测到最后一天之前的每一天预测的结果都只有第一天可用
        if not latest_date_predicted:
            pred_one.extend(pred[-1][1:])
            # 此时output_dates添加第一天以后日期
            output_dates.extend(
                [max_output_date + timedelta(d) for d in range(1, output_len)])
            dates_data.extend([dates_data[-1]] * (output_len - 1))
            if predicted_dates_to_delete:
                pgsql.delete_predicted_dates(detail_id,
                                             predicted_dates_to_delete)

        pred_detail_ids.append(detail_id)
        preds_one.append(pred_one)
        preds_all.append(pred)
        dates_pred_one.append(output_dates)
        dates_pred_data.append(dates_data)
        dates_pred_all.append(dates_pred_all_model)

    return preds_one, preds_all, dates_pred_one, dates_pred_all, dates_pred_data, pred_detail_ids, last_date_data_pred
