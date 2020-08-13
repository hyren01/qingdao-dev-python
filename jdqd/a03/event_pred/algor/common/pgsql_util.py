# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np
import feedwork.AppinfoConf as appconf
from feedwork.utils import UuidHelper
from jdqd.a03.event_pred.algor.common import preprocess as pp
from jdqd.a03.event_pred.bean.t_event_model import EventModel
from jdqd.a03.event_pred.bean.t_event_task import EventTask
from jdqd.a03.event_pred.bean.event_predict import EventPredict
from jdqd.a03.event_pred.enum.data_status import DataStatus
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
from feedwork.utils.DateHelper import sys_date, sys_time

# 读取配置文件
__cfg_data = appconf.appinfo["a03"]['data_source']
event_dbname = __cfg_data.get('event_dbname')  # 事件库数据库名
data_dbname = __cfg_data.get('data_dbname')   # 特征库数据库名

sys_date_formatter = '%Y-%m-%d'
sys_date_formatter_db = 'yyyy-MM-dd'
sys_time_formatter = '%H:%M:%S'


def update_model_status(model_id, status):
    """
    更新t_event_model表的模型状态。

    :param model_id: string. 模型编号
    :param status: string. 模型状态
    """
    sql = "UPDATE t_event_model SET status = %s WHERE model_id = %s"

    __modify(sql, (status, model_id))


def model_train_finish(model_id, status):
    """
    更新t_event_model表的模型状态。

    :param model_id: string. 模型编号
    :param status: string. 模型状态
    """
    sql = "UPDATE t_event_model SET status = %s, tran_finish_date = %s, tran_finish_time = %s WHERE model_id = %s"

    __modify(sql, (status, sys_date(sys_date_formatter), sys_time(sys_time_formatter), model_id))


def model_train_done_rnn(model_id, lag_dates, pcas, file_path_list, sub_model_names, outputs_list, events_set,
                         model_dir):
    """
    更新t_event_model_file、t_event_model_detail、t_event_model_tran表的数据。

    :param model_id: string. 模型编号
    :param file_path_list: array. 模型地址
    """
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        db.begin_transaction()

        sql = "INSERT INTO t_event_model_file(file_id, file_url, model_id) VALUES (%s, %s, %s) "
        params = []
        for model_fp in file_path_list:
            param = (UuidHelper.guid(), model_fp, model_id)
            params.append(param)
        db.executemany(sql, params)

        # 子模型信息入库
        detail_ids = []
        sql = "INSERT INTO t_event_model_detail(detail_id, model_name, status, model_id, lag_date, pca, create_date, " \
              "create_time) values (%s, %s, %s, %s, %s, %s, %s, %s)"
        params = []
        for sub_model_name, lag_date, pca in zip(sub_model_names, lag_dates, pcas):
            detail_id = UuidHelper.guid()
            detail_ids.append(detail_id)
            params.append((detail_id, sub_model_name, DataStatus.SUCCESS.value, model_id, int(lag_date), int(pca),
                           sys_date(sys_date_formatter), sys_time(sys_time_formatter)))
        db.executemany(sql, params)

        # 分事件模型信息入库
        sql = "INSERT INTO t_event_model_tran(tran_id, event_name, num, detail_id, status, create_date, create_time) " \
              "values (%s, %s, %s, %s, %s, %s, %s)"
        params = []
        for detail_id, outputs in zip(detail_ids, outputs_list):
            events_num = pp.get_event_num(outputs, events_set)
            for e in events_set:
                tran_id = UuidHelper.guid()
                event_num = events_num[e]
                param = (tran_id, e, event_num, detail_id, DataStatus.SUCCESS.value, sys_date(sys_date_formatter),
                         sys_time(sys_time_formatter))
                params.append(param)
        db.executemany(sql, params)

        sql = "UPDATE t_event_model SET model_dir = %s WHERE model_id = %s"
        db.execute(sql, (model_dir, model_id))

        db.commit()
        return detail_ids
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()


def model_train_done_cnn(model_id, kernel_size_array, pool_size_array, lag_date_array, file_path_array,
                         sub_model_name_array, output_array, event_set_array, model_dir):
    """
    更新t_event_model_file、t_event_model_detail、t_event_model_tran表的数据。

    :param model_id: string. 模型编号
    :param kernel_size_array: array. 卷积核列表
    :param pool_size_array: array. 过滤器列表
    :param lag_date_array: array. 滞后期列表
    :param file_path_array: array. 模型文件地址列表
    :param sub_model_name_array: array. 模型文件名列表
    :param output_array: array.事件样本列表（事件表数据）
    :param event_set_array: array.事件类别列表
    :param model_dir: str.模型存放地址
    :return array, 子模型编号列表
    """
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        db.begin_transaction()

        sql = "INSERT INTO t_event_model_file(file_id, file_url, model_id) VALUES (%s, %s, %s) "
        params = []
        for model_fp in file_path_array:
            param = (UuidHelper.guid(), model_fp, model_id)
            params.append(param)
        db.executemany(sql, params)

        # 子模型信息入库
        detail_ids = []
        sql = "INSERT INTO t_event_model_detail(detail_id, model_name, status, model_id, lag_date, kernel_size, " \
              "pool_size, create_date, create_time) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        params = []
        for sub_model_name, lag_date, kernel_size, pool_size in zip(sub_model_name_array, lag_date_array,
                                                                    kernel_size_array, pool_size_array):
            detail_id = UuidHelper.guid()
            detail_ids.append(detail_id)
            params.append((detail_id, sub_model_name, DataStatus.SUCCESS.value, model_id, int(lag_date),
                           int(kernel_size), int(pool_size), sys_date(sys_date_formatter),
                           sys_time(sys_time_formatter)))
        db.executemany(sql, params)

        # 分事件模型信息入库
        sql = "INSERT INTO t_event_model_tran(tran_id, event_name, num, detail_id, status, create_date, create_time) " \
              "values (%s, %s, %s, %s, %s, %s, %s)"
        params = []
        for detail_id, outputs in zip(detail_ids, output_array):
            events_num = pp.get_event_num(outputs, event_set_array)
            for event in event_set_array:
                tran_id = UuidHelper.guid()
                event_num = events_num[event]
                param = (tran_id, event, event_num, detail_id, DataStatus.SUCCESS.value, sys_date(sys_date_formatter),
                         sys_time(sys_time_formatter))
                params.append(param)
        db.executemany(sql, params)

        sql = "UPDATE t_event_model SET model_dir = %s WHERE model_id = %s"
        db.execute(sql, (model_dir, model_id))

        db.commit()
        return detail_ids
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()


def model_eval_done(top_scores, events_num):
    """
    更新t_event_model表的模型状态、训练开始日期、训练结束日期。

    :param top_scores:
    :param events_num:
    """
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        sql = "insert into t_event_model_tot(tot_id, num, false_rate, recall_rate, false_alarm_rate, " \
              "tier_precision, tier_recall, bleu, score, status, detail_id, create_date, create_time) values " \
              "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        params = []
        for score, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary, \
                fa_summary, detail_id in top_scores:
            num_events = np.sum([v for k, v in events_num.items() if str(k) != '0'])
            param = (UuidHelper.guid(), str(num_events), str(fr_summary), str(rc_summary), str(fa_summary),
                     str(tier_precision_summary), str(tier_recall_summary), str(bleu_summary), str(score),
                     DataStatus.SUCCESS.value, detail_id, sys_date(sys_date_formatter), sys_time(sys_time_formatter))
            params.append(param)
        db.executemany(sql, params)
        db.commit()
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


def insert_into_model_detail(sub_model_names, model_id):
    """
    将子模型信息入库
    Args:
      sub_model_names: 子模型名称, 形如 model_name-input_len-output_len-n_pca
      model_id: 模型编号

    Returns:
        array，detail_id
    """
    detail_ids = []
    sql = "INSERT INTO t_event_model_detail(detail_id, model_name, status, model_id, create_date, create_time) " \
          "values (%s, %s, %s, %s, %s, %s)"
    params = []
    for sub_model_name in sub_model_names:
        detail_id = UuidHelper.guid()
        detail_ids.append(detail_id)
        params.append((detail_id, sub_model_name, DataStatus.SUCCESS.value, model_id, sys_date(sys_date_formatter),
                       sys_time(sys_time_formatter)))

    __modify_many(sql, params)

    return detail_ids


def insert_model_test(event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
                      detail_id):
    """
    记录模型评估信息
    Args:
        event: 事件名称
        event_num: 事件在测试(评估)数据集中出现的次数
        false_rate: 误报率
        recall_rate: 召回率
        false_alarm_rate: 虚警率
        tier_precision: 头部精确率, 即将预测值降序排序后, 头部(前n个)预测值中预测正确的结果数所占的比例
        tier_recall: 头部召回率, 即将预测值降序排序后, 头部(前n个)预测值中预测正确的结果数与真实正例个数的比值
        bleu: bleu指标
        detail_id: 子模型对应的id
    """
    test_id = UuidHelper.guid()
    sql = "insert into t_event_model_test(test_id, event_name, num, false_rate, recall_rate, false_alarm_rate, " \
          "tier_precision, tier_recall, bleu, status, detail_id, create_date, create_time) " \
          "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    param = (test_id, event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
             DataStatus.SUCCESS.value, detail_id, sys_date(sys_date_formatter), sys_time(sys_time_formatter))

    __modify(sql, param, '插入子模型评估结果失败')


def query_sub_models_by_model_id(model_id):
    """
    根据模型编号查询出综合排名前10的子模型。
    :param model_id:  模型编号
    :return array. t_event_model_detail表数据
    """

    db = DatabaseWrapper(dbname=event_dbname)
    try:
        sql = "SELECT t1.model_name, t1.detail_id, t1.lag_date, t1.pca, t1.kernel_size, t1.pool_size, t2.days, " \
              "t2.model_dir FROM t_event_model_detail t1 JOIN t_event_model t2 ON t1.model_id = t2.model_id " \
              "JOIN t_event_model_tot t3 ON t1.detail_id = t3.detail_id " \
              "WHERE t1.model_id = %s ORDER BY t3.score DESC LIMIT 10"
        event_predict = db.query(sql, (model_id,), result_type=QueryResultType.BEAN, wild_class=EventPredict)
        if len(event_predict) < 1:
            raise RuntimeError(f"Query t_event_model_detail error, the {model_id} cannot get multi-row")

        return event_predict
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


def query_predicted_rsts(detail_id, pred_start_date, task_id):
    """根据子模型的 detail_id 列表以及开始预测日期查询 detail_id 对应子模型自开始预测日期之后
    的已预测日期
    Args:
      detail_id:
      pred_start_date:
      task_id:

    Returns: dict. key: detail_id, value: 已预测的日期列表
    """
    sql = "select detail_id, forecast_date from t_event_task_rs WHERE task_id = %s and detail_id = %s " \
          "and forecast_date >= %s"
    results = __query(sql, event_dbname, (task_id, detail_id, pred_start_date))
    detail_id_dates = {}
    for r in results:
        detail_id_dates.setdefault(r[0], set()).add(r[1])

    return detail_id_dates


def delete_predicted_dates(detail_id, dates_str):
    """删除子模型已预测日期的预测结果
    Args:
      detail_id: 子模型 id
      dates_str: 需删除的已预测的日期列表, 元素类型为 str
    """
    sql = "delete from t_event_task_rs where detail_id = %s and forecast_date = %s"
    params = []
    for date in dates_str:
        params.append((detail_id, str(date)))

    __modify_many(sql, params)


def insert_pred_result(probs, probs_all_days, dates, dates_pred_all, dates_data, detail_ids, events_set, task_id):
    """向数据库插入预测结果
    Args:
      probs: 预测结果
      probs_all_days: 多天预测结果
      dates:
      dates_pred_all: 多天预测日期
      dates_data: 数据日期
      detail_ids:
      events_set:
      task_id:
    """
    sql_task_rs = "insert into t_event_task_rs(rs_id, event_name, probability, forecast_date, status, " \
                  "detail_id, task_id, create_date, create_time, predict_end_date) " \
                  "values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    sql_task_rs_his = "insert into t_event_task_rs_his(rs_id, event_name, probability, forecast_date," \
                      "detail_id, task_id, create_date, create_time, predict_end_date) values " \
                      "(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    sql_task_rs_params = []
    sql_task_rs_his_params_hist = []
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        db.begin_transaction()
        for p, pa, ds, da, did, dd in zip(probs, probs_all_days, dates, dates_pred_all, detail_ids, dates_data):
            for i, e in enumerate(events_set):
                for j, d in enumerate(ds):
                    param = (UuidHelper.guid(), str(e), f'{p[j][i]:.4f}', str(d), DataStatus.SUCCESS.value, did,
                             task_id, sys_date(sys_date_formatter), sys_time(sys_time_formatter), str(dd[j]))
                    sql_task_rs_params.append(param)
                for j, d in enumerate(da):
                    pd = pa[j]
                    for k, d_ in enumerate(d):
                        pd_ = pd[k]
                        param_hist = (UuidHelper.guid(), str(e), f'{pd_[i]:.4f}', str(d_), did, task_id,
                                      sys_date(sys_date_formatter), sys_time(sys_time_formatter), str(dd[j]))
                        sql_task_rs_his_params_hist.append(param_hist)
            db.executemany(sql_task_rs, sql_task_rs_params)
            sql_task_rs_params = []
            db.executemany(sql_task_rs_his, sql_task_rs_his_params_hist)
            sql_task_rs_his_params_hist = []
        db.commit()     # 配置文件中设置不自动提交，所以手动提交
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()


def update_task_status(task_id, status):
    """
    修改t_event_task表的运行状态。
    :param task_id: string. 任务编号
    :param status: string. 任务运行状态
    """
    sql = f"UPDATE t_event_task SET status = %s WHERE task_id = %s"
    param = (status, task_id)
    __modify(sql, param)


def predict_task_finish(task_id, date_data_pred, status):
    """
    修改t_event_task表的运行状态、任务结束日期、预测结束日期、任务结束事件。
    :param task_id: string. 任务编号
    :param date_data_pred: string. 预测结束日期
    :param status: string. 任务运行状态
    """
    sql = "UPDATE t_event_task SET status = %s, task_finish_date = %s, predict_end_date = %s, task_finish_time = %s " \
          "WHERE task_id = %s"
    param = (status, sys_date(sys_date_formatter), str(date_data_pred), sys_time(sys_time_formatter), task_id)
    __modify(sql, param)


def query_data_table_2pandas(table_name):
    """
    查询数据库中指定数据表的所有数据。该方法会将查询到的数据中date_col字段数据转换为字符串类型。

    :param table_name: string. 数据表名
    :return dataframe.指定表的dataframe形式。
    """
    # 每次查询建立一次连接
    db = DatabaseWrapper(dbname=data_dbname)
    try:
        sql = f"SELECT * FROM {table_name} ORDER BY rqsj ASC"
        result = db.query(sql, (), result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


def query_event_table_2pandas(table_name, event_col, date_col):
    """
    查询数据库中指定事件表的所有数据，若传入了not_none_columns参数，则该方法不会返回指定参数中的列名在数据表为空的行。

    :param table_name: string. 事件表名
    :param event_col: string. 事件类别字段名。
    :param date_col: string. 事件日期字段名。
    :return dataframe.指定表的dataframe形式。
    """
    # 每次查询建立一次连接
    db = DatabaseWrapper(dbname=data_dbname)
    try:
        sql = f"SELECT {event_col}, CAST({date_col} AS VARCHAR) AS {date_col} FROM {table_name} " \
            f"WHERE qssj IS NOT NULL AND ({event_col} IS NOT NULL AND {event_col} <> '') AND " \
            f"({date_col} IS NOT NULL AND CAST({date_col} AS VARCHAR) <> '') AND " \
            f"LENGTH(CAST({date_col} AS VARCHAR)) > 8 ORDER BY {date_col} ASC"
        result = db.query(sql, (), result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


def __query(sql, db: str = data_dbname, parameter: tuple = ()):
    """
    从数据库查询数据
    Args:
      sql:
      parameter:
      db: 连接的数据库名称. 'alg' 为算法所需源数据所在库, 如需指定算法生成数据及应用端数据
      所在库, 将此参数指定为其他名称即可, 推荐使用 'mng'

    Returns:
      查询结果
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")
    db = DatabaseWrapper(dbname=db)
    try:
        result = db.query(sql, parameter, QueryResultType.DB_NATURE)
        result = [list(r.values()) for r in result]
        return result
    except Exception as e:
        raise RuntimeError(f"The query error! {e}")
    finally:
        db.close()


def query_teventmodel_by_id(model_id):
    """
    根据模型编号查询t_event_model表数据。若该模型编号查询出的数据行数不为1则抛出RuntimeError异常。

    :param model_id: string. 模型编号
    :return EventModel.实体对象。
    """
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        sql = "SELECT * FROM t_event_model WHERE model_id = %s"
        t_event_model = db.query(sql, (model_id,), result_type=QueryResultType.BEAN, wild_class=EventModel)
        if len(t_event_model) != 1:
            raise RuntimeError(f"Query t_event_model error, the {model_id} cannot get only one row")
        return t_event_model[0]
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


def query_teventtask_by_id(task_id):
    """
    根据模型编号查询t_event_task表数据。若该模型编号查询出的数据行数不为1则抛出RuntimeError异常。

    :param task_id: string. 预测任务编号
    :return EventModel.实体对象。
    """
    db = DatabaseWrapper(dbname=event_dbname)
    try:
        sql = "SELECT t2.event_type, t2.model_type, t2.model_dir, t2.event, t1.* FROM t_event_task t1 " \
              "JOIN t_event_model t2 ON t1.model_id = t2.model_id WHERE t1.task_id = %s"
        t_event_task = db.query(sql, (task_id,), result_type=QueryResultType.BEAN, wild_class=EventTask)
        if len(t_event_task) != 1:
            raise RuntimeError(f"Query t_event_task error, the {task_id} cannot get only one row")
        return t_event_task[0]
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


def __modify_many(sql: str, parameters: list, error=''):
    """
    根据 sql 对数据进行增删改操作
    Args:
      sql: str. sql字符串
      parameters: array. 参数列表，用于支持多批量执行
      error: 操作出错时则日志中输出的错误信息
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")

    db = DatabaseWrapper(dbname=event_dbname)
    try:
        db.executemany(sql, parameters)
        db.commit()     # 配置文件中设置不自动提交，所以手动提交
    except Exception as e:
        raise RuntimeError(f'{error}: {str(e)} ' if error else error)
    finally:
        db.close()


def __modify(sql: str, parameter: tuple, error=''):
    """
    根据 sql 对数据进行增删改操作
    Args:
      sql: str. sql字符串
      parameter: tuple. 参数列表
      error: 操作出错时则日志中输出的错误信息
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")

    db = DatabaseWrapper(dbname=event_dbname)
    try:
        db.execute(sql, parameter)
        db.commit()  # 配置文件中设置不自动提交，所以手动提交
    except Exception as e:
        raise RuntimeError(f'{error}: {str(e)} ' if error else error)
    finally:
        db.close()
