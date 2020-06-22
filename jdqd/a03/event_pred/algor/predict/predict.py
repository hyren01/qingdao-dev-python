# -*- coding: utf-8 -*-
import numpy as np
from datetime import timedelta

from jdqd.a03.event_pred.algor.train import model_evalution as mu
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, \
  preprocess as pp
from feedwork.utils import logger
import feedwork.AppinfoConf as appconf
from feedwork.utils.FileHelper import cat_path


class PredictModel(object):

    def __init__(self, model_dir, load_model=True):
        """预测模型初始化
        Args:
          model_dir: 存放模型的目录名
          load_model: 是否加载已有模型
        """
        if load_model:
            self.encoder, self.decoder = mu.load_models(model_dir)

    def predict_sample(self, encoder, decoder, input_sample, n_classes, output_len):
        """
        单个样本预测
        Args:
          encoder: encoder 模型
          decoder: decoder 模型
          input_sample: encoder 模型的输入样本
          n_classes: 事件类别个数
          output_len: decoder 输出长度, 即预测天数

        Returns:
          单个样本的预测结果, shape(1, 预测天数, 事件类别数)
        """
        state = encoder.predict(np.array([input_sample]))
        target_seq = np.array([0.0 for _ in range(n_classes)]).reshape(
          [1, 1, n_classes])
        output = []
        for t in range(output_len):
            yhat, h, c = decoder.predict([target_seq] + state)
            output.append(yhat[0, 0, :])
            state = [h, c]
            target_seq = yhat
        return output

    def pred_with_reloaded_model(self, inputs, n_classes, output_len):
        """使用加载的模型预测测试集数据
        Args:
          inputs: 预测输入数据, 为降维后的数据表数据, shape(样本数, 降维维度)
          n_classes: 事件类别数
          output_len: decoder 输出长度, 即预测天数

        Returns:
          输入样本的预测结果, shape(样本数, 预测天数, 事件类别)
        """
        preds = [self.predict_sample(self.encoder, self.decoder, inputs_sample, n_classes, output_len)
                 for inputs_sample in inputs]
        return preds


def predict(model_dir, inputs, output_len, n_classes):
    """
    执行页面的预测请求
    Args:
      model_dir: 模型文件存放路径
      inputs: 预测输入数据, 为降维后的数据表数据, shape(样本数, 降维维度)
      n_classes: 事件类别数
      output_len: decoder 输出长度, 即预测天数

    Returns:
        输入样本的预测结果, shape(样本数, 预测天数, 事件类别)
    """
    predict_model = PredictModel(model_dir)
    preds = predict_model.pred_with_reloaded_model(inputs, n_classes, output_len)
    return preds


models_dir = cat_path(appconf.ALGOR_MODULE_ROOT, 'event_pred')


def predict_by_sub_models(data, dates, detail_ids, sub_models, pred_start_date, num_classes, task_id):
    """

    Args:
      data: 预测输入数据
      dates: 数据表日期列表
      detail_ids: 子模型的 detail_id 列表
      sub_models: 子模型的 存放路径列表
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
    predicted_detail_id_dates = pgsql.query_predicted_rsts(detail_ids, pred_start_date, task_id)
    for detail_id, sub_model in zip(detail_ids, sub_models):
        logger.info(f'正在使用模型{sub_model}进行预测')
        params = sub_model.split('-')[-3:]
        input_len, output_len, n_pca = [int(p) for p in params]
        model_dir = cat_path(models_dir, sub_model)
        values_pca = pp.apply_pca(n_pca, models_dir, data, True)
        inputs_test, output_dates = pp.gen_inputs_by_pred_start_date(values_pca, input_len, dates, pred_start_date)
        max_output_date = output_dates[-1] + timedelta(1)
        output_dates.append(max_output_date)  # 此时output_dates不包含预测第一天后日期
        dates_data = [output_dates[0] - timedelta(1)]
        dates_data.extend(output_dates[:-1])

        last_date_data_pred = dates_data[-1]

        predicted_dates = predicted_detail_id_dates.get(detail_id)  # type of list of str
        if predicted_dates is None:
            latest_date_predicted = False
            predicted_dates_to_delete = []
        else:
            predicted_dates = sorted([pp.parse_date_str(d) for d in predicted_dates])
            predicted_dates_to_delete = predicted_dates[-output_len + 1:]
            predicted_dates = predicted_dates[:-output_len + 1]  # 截取只预测一天的预测结果
            max_predicted_date = predicted_dates[-1]
            zipped_unpredicted = [[d, i, dd] for d, i, dd in zip(output_dates, inputs_test, dates_data)
                                  if d not in predicted_dates]
            if not zipped_unpredicted:
                logger.info(f'{sub_model}所有日期已预测, 跳过')
                continue

            output_dates, inputs_test, dates_data, = zip(*zipped_unpredicted)
            output_dates = list(output_dates)
            inputs_test = list(inputs_test)
            dates_data = list(dates_data)
            if max_predicted_date == max_output_date:
                latest_date_predicted = True
            else:
                latest_date_predicted = False

        # 预测日期, 包含第一天以后日期
        dates_pred_all_model = [[dd + timedelta(t) for t in range(1, output_len + 1)] for dd in dates_data]

        pred = predict(model_dir, inputs_test, output_len, num_classes)

        pred_one = [p[0] for p in pred]  # 只预测一天
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
