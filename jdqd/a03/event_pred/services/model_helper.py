# -*- coding:utf-8 -*-
import os
import feedwork.AppinfoConf as appconf
from feedwork.utils import logger
from jdqd.a03.event_pred.algor.predict import predict
from jdqd.a03.event_pred.algor.train.train_model import SeqModel
from jdqd.a03.event_pred.algor.common import preprocess
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, preprocess as pp, obtain_data as od
from jdqd.a03.event_pred.algor.train import model_evalution as meval
from jdqd.a03.event_pred.enum.event_type import EventType
from jdqd.a03.event_pred.enum.model_status import ModelStatus
from jdqd.a03.event_pred.enum.event_evalution_status import ModelEvalutionStatus
from feedwork.utils.DateHelper import sys_date, sys_time
from feedwork.utils.FileHelper import cat_path

# 读取配置文件
__cfg_data = appconf.appinfo["a03"]['data_source']

super_event_type_col = __cfg_data.get('super_event_type_col')   # 大类事件字段名
sub_event_type_col = __cfg_data.get('sub_event_type_col')   # 小类事件字段名
event_table_name = __cfg_data.get('event_table_name')   # 用于训练及预测，在数据库中的数据表名（事件表）

date_col = __cfg_data.get('date_col')      # 时间字段名
event_priority = __cfg_data['event_priority']
# 模型存放路径
models_dir = cat_path(appconf.ALGOR_MODULE_ROOT, 'event_pred')


def transform_data(dates, model_id):
    event_type = pgsql.query_event_type_by_id(model_id)
    event_col = super_event_type_col if event_type == EventType.SUB_EVENT_TYPE.value else sub_event_type_col  # '2' 为大事件
    # TODO 同时写清楚补0之前数据格式及补0后数据格式
    # 补 0 后的事件列表
    events_p = od.get_events(event_table_name, dates, event_priority, event_col=event_col, date_col=date_col)
    events_set = pp.get_events_set(events_p)  # 事件集
    events_p_oh = pp.events_one_hot(events_p, events_set)  # 补 0 后 one-hot 形式的事件列表

    return events_set, events_p_oh


def train_over_hyperparameters(model_id, data, dates, events_set, events_p_oh, model_name, start_date,
                               end_date, output_len, min_pca_dim, max_pca_dim, min_input_len, max_input_len, step,
                               num_units=128, batch=64, epoch=150):
    """
    执行页面的训练模型请求, 遍历不同超参数的组合来训练模型. 先对降维维度遍历,
    再对encoder输入长度遍历, 产生的子模型数量为降维维度遍历个数与encoder输入长度遍历个数之积.
    随encoder输入长度, 降维维度的增加, 模型训练时间会变长. epoch越大, 模型训练时间越长
    Args:
      model_id: 模型目录
      data: 模型目录
      dates: 模型目录
      events_set: 模型目录
      events_p_oh: 模型目录
      model_name: 训练任务模型名称
      start_date: 开始预测日期
      end_date: 预测截止日期
      output_len: decoder 输出序列长度, 即一个样本的预测天数
      min_pca_dim: 最小降维维度
      max_pca_dim: 最大降维维度
      min_input_len: 最小 encoder 输入长度
      max_input_len: 最大 encoder 输入长度
      step: 遍历降维维度时选择的步长
      num_units: LSTM 单元隐藏层的维度
      batch: 训练每个批的样本数
      epoch: 遍历所有样本的训练次数.

    Returns:
      训练完成的模型文件所在目录列表
      模型名称列表
      每个模型对应的 decoder 输出序列列表. 不同的模型由于输入序列长度不同导致输出序列不同
      每个模型对应的超参数列表
    """
    logger.info('开始训练模型')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    pca_dir = models_dir
    sub_model_dirs = []
    sub_model_names = []
    outputs_list = []
    params_list = []
    try:
        # TODO 需要考虑训练中断后继续训练吗？ 降维和滞后期是如何选择的要写清楚
        # 先进行降维，对于PCA降维来说它始终只有固定的降维组合，若先对滞后期的选择的话，会导致出现N个重复的降维组合
        for i in range(min_pca_dim, max_pca_dim, step):  # 各种降维选择
            values_pca = preprocess.apply_pca(i, pca_dir, data)
            # 基于页面选择的开始日期、结束日期的整个范围中每一天作为一个基准日期，在该基准日期往前推max_input_len至min_input_len天
            # 的范围内每次间隔5天（10、15、20天）拉取数据训练模型。
            for j in range(min_input_len, max_input_len, 5):  # 滞后期的选择
                # TODO SeqModel类可以去掉，里面的方法做成静态的方式
                seq_model = SeqModel(j, output_len, num_units, batch, epoch, i)
                sub_model_name = f'{model_name}-{seq_model.n_in}-{seq_model.n_out}-{seq_model.pca_n}'
                sub_model_names.append(sub_model_name)
                sub_model_dir = cat_path(models_dir, sub_model_name)
                if not os.path.exists(sub_model_dir):
                    os.mkdir(sub_model_dir)
                sub_model_dirs.append(sub_model_dir)
                array_x, array_y, array_yin = seq_model.gen_samples(values_pca, events_p_oh, j, output_len, dates,
                                                                    start_date, end_date)
                outputs_list.append(array_y)
                params_list.append([j, output_len, i])
                seq_model.train(array_x, array_y, array_yin, sub_model_dir)

        logger.info('训练完成, 模型存入数据库')

        # TODO 下面的一系列数据库操作不需要事务？
        pgsql.model_train_done(model_id, sub_model_dirs)
        # 子模型信息入库
        detail_ids = pgsql.insert_into_model_detail(sub_model_names, model_id)
        # 分事件模型信息入库
        pgsql.insert_into_model_train(detail_ids, outputs_list, events_set, ModelStatus.SUCCESS.value)


        return sub_model_dirs, params_list, detail_ids
    except Exception as e:
        # TODO 其实update_model_status最后的调用会捕捉异常并抛出，如果这里没特别操作的话就不要再捕捉了？
        pgsql.update_model_status(model_id, ModelStatus.FAILD.value)
        raise RuntimeError(e)


def evaluate_sub_models(model_id, data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set,
                        eval_start_date, eval_end_date):
    logger.info('开始评估模型')
    n_classes = len(events_set)
    # 评估模型. scores: 子模型综合评分列表; events_num: 测试机事件个数
    scores, events_num = meval.evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh,
                                                   events_set, n_classes, eval_start_date, eval_end_date)
    # TODO top模型的筛选和入库是两个主要步骤，不应该混在一起？
    logger.info('模型评估结束, 筛选top模型')
    pgsql.insert_model_tot(scores, events_num)
    date_str = sys_date('%Y-%m-%d')
    time_str = sys_time('%H:%M:%S')
    pgsql.model_eval_done(model_id, date_str, time_str, ModelStatus.SUCCESS.value)


def web_predict(model_id, data, dates, events_set, tables, task_id, pred_start_date):
    """
    # TODO 这个注释跟没写一样，下面的参数没更新
    通过页面传入的参数使用指定模型进行预测, 并将预测结果存入数据库.
    Args:
      model_id: 预测使用的模型 id
      data: 预测使用的模型 id
      dates: 预测使用的模型 id
      events_set: 预测使用的模型 id
      tables: 预测使用的数据表列表
      task_id: 预测任务的 id
      pred_start_date: 开始预测日期, 即预测结果由此日期开始
    """
    try:
        sub_model_results = pgsql.query_sub_models_by_model_id(model_id)
        # 下标0: 子模型名称. 下标1: 子模型对应的 detail_id
        sub_models = [r[0] for r in sub_model_results]
        detail_ids = [r[1] for r in sub_model_results]
        # 事件类别数量(含0事件)
        num_classes = len(events_set)

        preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred, pred_detail_ids, last_date_data_pred = \
            predict.predict_by_sub_models(data, dates, detail_ids, sub_models, pred_start_date, num_classes, task_id)
        if pred_detail_ids:
            pgsql.insert_pred_result(preds, preds_all_days, dates_pred, dates_pred_all, dates_data_pred,
                                     pred_detail_ids, events_set, task_id)
        date_str = sys_date('%Y-%m-%d')
        time_str = sys_time('%H:%M:%S')
        pgsql.predict_task_done(task_id, date_str, time_str, last_date_data_pred, ModelStatus.SUCCESS.value)
        logger.info(f"当前表 {','.join(tables)} 的模型预测完成")
    except Exception as e:
        pgsql.update_task_status(task_id, ModelStatus.FAILD.value)
        raise RuntimeError(e)
