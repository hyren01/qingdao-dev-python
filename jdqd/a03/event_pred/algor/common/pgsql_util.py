# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import time
import numpy as np
from feedwork.utils import logger, UuidHelper
from jdqd.a03.event_pred.algor.common import preprocess as pp
from jdqd.a03.event_pred.bean.t_event_model import EventModel
from feedwork.database.bean.database_config import DatabaseConfig
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType


def update_model_status(model_id, status):
    """
    更新t_event_model表的模型状态。

    :param model_id: string. 模型编号
    :param status: string. 模型状态
    """
    sql = "UPDATE t_event_model SET status = %s WHERE model_id = %s"
    __modify(sql, [(status, model_id)])


def model_train_done(model_id, file_path_list):
    """
    更新t_event_model_file表的数据。

    :param model_id: string. 模型编号
    :param file_path_list: array. 模型地址
    """
    sqls = []
    params = []
    for model_fp in file_path_list:
        sql = "INSERT INTO t_event_model_file(file_id, file_url, model_id) VALUES(%s, %s, %s) "
        sqls.append(sql)
        uuid = UuidHelper.guid()
        param = (uuid, model_fp, model_id)
        params.append(param)
    __modify(sqls, params)


def model_eval_done(model_id, date_str, time_str, status):
    """
    更新t_event_model表的模型状态、训练开始日期、训练结束日期。

    :param model_id: string. 模型编号
    :param date_str: string. 训练开始日期
    :param time_str: string. 训练结束日期
    :param status: string. 模型状态
    """
    sql = "UPDATE t_event_model SET status = %s, tran_finish_date = %s, tran_finish_time = %s WHERE model_id = %s"
    param = [(status, date_str, time_str, model_id)]
    __modify(sql, param)


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
    status = 1
    sqls = []
    params = []
    for sub_model_name in sub_model_names:
        detail_id = UuidHelper.guid()
        detail_ids.append(detail_id)
        sql = "INSERT INTO t_event_model_detail(detail_id, model_name, status, " \
              "model_id) values(%s, %s, %s, %s)"
        sqls.append(sql)
        param = (detail_id, sub_model_name, status, model_id)
        params.append(param)
    __modify(sqls, params)

    return detail_ids


def insert_into_model_train(detail_ids, outputs_list, events_set, status):
    """
    记录模型训练信息
    Args:
      detail_ids: array.detail_id
      outputs_list: array.输出序列
      events_set: array.去重后的事件类别
      status: string.模型训练状态
    """
    sqls = []
    params = []
    for detail_id, outputs in zip(detail_ids, outputs_list):
        events_num = pp.get_event_num(outputs, events_set)
        for e in events_set:
            tran_id = UuidHelper.guid()
            event_num = events_num[e]
            sql = "INSERT INTO t_event_model_tran(tran_id, event_name, num, detail_id, status) values " \
                  "(%s, %s, %s, %s, %s)"
            sqls.append(sql)
            param = (tran_id, e, event_num, detail_id, status)
            params.append(param)
    __modify(sqls, params)


def insert_model_test(event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
                      status, detail_id):
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
        status: 模型评估状态
        detail_id: 子模型对应的id
    """
    test_id = UuidHelper.guid()
    sql = "insert into t_event_model_test(test_id, event_name, num, false_rate, recall_rate, false_alarm_rate, " \
          "tier_precision, tier_recall, bleu, status, detail_id)values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    param = [(test_id, event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
              status, detail_id)]
    error = '插入子模型评估结果失败'
    __modify(sql, param, error)


def insert_model_tot(scores, events_num):
    """
    将模型评估结果插入数据库
    :param scores:  评估分数
    :param events_num:  事件类别数
    """
    status = '1'
    scores.sort(key=lambda x: x[0], reverse=True)
    top_scores = scores[:min(10, len(scores))]
    logger.info('top模型存入数据库')
    sqls = []
    params = []
    for score, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary, fa_summary, \
            detail_id in top_scores:
        num_events = np.sum([v for k, v in events_num.items() if str(k) != '0'])
        tot_id = UuidHelper.guid()
        sql_summary = "insert into t_event_model_tot(tot_id, num, false_rate, recall_rate, false_alarm_rate, " \
                      "tier_precision, tier_recall, bleu, score, status, detail_id) values " \
                      "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        sqls.append(sql_summary)
        param = (tot_id, str(num_events), str(fr_summary), str(rc_summary), str(fa_summary),
                 str(tier_precision_summary), str(tier_recall_summary), str(bleu_summary), str(score), status,
                 detail_id)
        params.append(param)
    __modify(sqls, params, '存入top模型出错')


def query_sub_models_by_model_id(model_id):
    """
    根据模型编号查询出子模型。
    :param model_id:  模型编号
    :return array. t_event_model_detail表数据
    """
    sql = "SELECT detail_id, score FROM t_event_model_tot WHERE detail_id IN " \
          "(SELECT detail_id FROM t_event_model_detail WHERE model_id = %s) "
    # TODO 数据库名是写死的，且不是统一在配置文件中配置的
    results1 = __query(sql, 'mng', (model_id,))
    detail_ids = [r[0] for r in results1]
    detail_ids = tuple(detail_ids)
    sql2 = "SELECT model_name, detail_id FROM t_event_model_detail WHERE detail_id IN %s"
    results2 = __query(sql2, 'mng', (detail_ids,))

    return results2


def query_event_type_by_id(model_id):
    """
    根据模型编号查询该模型使用的事件类型（大类、小类）。当查询的数据行数不唯一则抛出RuntimeError异常。
    :param model_id:  模型编号
    :return string. t_event_model_detail表数据
    """
    sql = "SELECT event_type FROM t_event_model WHERE model_id = %s"
    result = __query(sql, 'mng', (model_id,))
    if len(result) != 1:
        raise RuntimeError(f"Query error, model_id cannot get a row {model_id}")
    # db_conn.query返回的是[['xxx']]
    return result[0][0]


def query_predicted_rsts(detail_ids, pred_start_date, task_id):
    """根据子模型的 detail_id 列表以及开始预测日期查询 detail_id 对应子模型自开始预测日期之后
    的已预测日期
    Args:
      detail_ids:
      pred_start_date:
      task_id:

    Returns: dict. key: detail_id, value: 已预测的日期列表
    """
    sql = "select detail_id, forecast_date from t_event_task_rs where task_id = %s and detail_id in %s " \
          "and forecast_date >= %s"
    param = (task_id, tuple(detail_ids), pred_start_date)
    results = __query(sql, 'mng', param)
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
    sqls = []
    params = []
    for d in dates_str:
        sql = f"delete from t_event_task_rs where detail_id = %s and forecast_date = %s"
        params.append((detail_id, str(d)))
        sqls.append(sql)
    __modify(sqls, params)


def insert_pred_result(probs, probs_all_days, dates, dates_pred_all, dates_data,
                       detail_ids,
                       events_set, task_id):
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
    sqls = []
    params = []
    sqls_hist = []
    params_hist = []
    for p, pa, ds, da, did, dd in zip(probs, probs_all_days, dates, dates_pred_all, detail_ids, dates_data):
        for i, e in enumerate(events_set):
            for j, d in enumerate(ds):
                rs_id = UuidHelper.guid()
                date_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime(time.time()))
                date_str, time_str = date_time.split(' ')
                sql_task_rs = "insert into t_event_task_rs(rs_id, event_name, " \
                              "probability, forecast_date, status, detail_id, " \
                              "task_id, create_date, create_time, predict_end_date) " \
                              "values(%s, %s, %s, %s, '1', %s, %s, %s, %s, %s)"
                sqls.append(sql_task_rs)
                param = (rs_id, str(e), f'{p[j][i]:.4f}', str(d), did, task_id,
                         date_str, time_str, str(dd[j]))
                params.append(param)
            for j, d in enumerate(da):
                pd = pa[j]

                for k, d_ in enumerate(d):
                    pd_ = pd[k]

                    rs_id = UuidHelper.guid()
                    date_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.localtime(time.time()))
                    date_str, time_str = date_time.split(' ')
                    sql_task_rs_his = "insert into t_event_task_rs_his(rs_id, " \
                                      "event_name, probability, forecast_date, " \
                                      "detail_id, task_id, create_date, create_time, " \
                                      "predict_end_date) values(%s, %s, %s, %s, %s, " \
                                      "%s, %s, %s, %s)"
                    sqls_hist.append(sql_task_rs_his)
                    param_hist = (rs_id, str(e), f'{pd_[i]:.4f}', str(d_), did, task_id,
                                  date_str, time_str, str(dd[j]))
                    params_hist.append(param_hist)
        error = f't_event_task_rs表预测结果插入出错'
        __modify(sqls, params, error)

        error_his = f't_event_task_rs_his表预测结果插入出错'
        __modify(sqls_hist, params_hist, error_his)


def update_task_status(task_id, status):
    """
    修改t_event_task表的运行状态。
    :param task_id: string. 任务编号
    :param status: string. 任务运行状态
    """
    sql = f"UPDATE t_event_task SET status = %s WHERE task_id = %s"
    param = (status, task_id)
    __modify(sql, [param])


def predict_task_done(task_id, date_str, time_str, date_data_pred, status):
    """
    修改t_event_task表的运行状态、任务结束日期、预测结束日期、任务结束事件。
    :param task_id: string. 任务编号
    :param date_str: string. 任务结束日期
    :param time_str: string. 任务结束时间
    :param date_data_pred: string. 预测结束日期
    :param status: string. 任务运行状态
    """
    sql = "UPDATE t_event_task SET status = %s, task_finish_date = %s, predict_end_date = %s, task_finish_time = %s " \
          "WHERE task_id = %s"
    param = (status, date_str, str(date_data_pred), time_str, task_id)
    __modify(sql, [param])


# def query_table(table_name):
#     """
#     查询指定表的所有数据。
#
#     :param table_name: string. 数据表名
#     :return array.二维数组。
#     """
#     # 每次查询建立一次连接
#     database_config = DatabaseConfig()
#     database_config.name = 'alg'
#     db = DatabaseWrapper(database_config)
#     try:
#         sql = f"SELECT * FROM {table_name}"
#         result = db.query(sql, (), result_type=QueryResultType.DB_NATURE)
#         result = [list(r.values()) for r in result]
#         return result
#     except Exception as e:
#         raise RuntimeError(f"Query table error! {str(e)}")
#     finally:
#         # db不可能为None
#         db.close()


def query_data_table_2pandas(table_name, date_col):
    """
    查询数据库中指定数据表的所有数据。该方法会将查询到的数据中date_col字段数据转换为字符串类型。

    :param table_name: string. 数据表名
    :param date_col: string. 数据表中日期字段名
    :return dataframe.指定表的dataframe形式。
    """
    # 每次查询建立一次连接
    database_config = DatabaseConfig()
    database_config.name = 'alg'
    db = DatabaseWrapper(database_config)
    try:
        sql = f"SELECT CAST({date_col} AS VARCHAR) AS {date_col},* FROM {table_name}"
        result = db.query(sql, (), result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


def query_event_table_2pandas(table_name):
    """
    查询数据库中指定事件表的所有数据。

    :param table_name: string. 事件表名
    :return dataframe.指定表的dataframe形式。
    """
    # 每次查询建立一次连接
    database_config = DatabaseConfig()
    database_config.name = 'alg'
    db = DatabaseWrapper(database_config)
    try:
        sql = f"SELECT * FROM {table_name}"
        result = db.query(sql, (), result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(f"Query table error! {str(e)}")
    finally:
        # db不可能为None
        db.close()


def __query(sql, db: str = 'alg', parameter: tuple = ()):
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
    database_config = DatabaseConfig()
    database_config.name = db
    db = DatabaseWrapper(database_config)
    try:
        result = db.query(sql, parameter, QueryResultType.DB_NATURE)
        result = [list(r.values()) for r in result]
        return result
    except Exception as e:
        raise RuntimeError(f"The query error! {e}")
    finally:
        db.close()


def __modify(sql, parameters=(), error=''):
    """
    根据 sql 对数据进行增删改操作
    Args:
      sql:
      parameters:
      error: 操作出错时则日志中输出的错误信息
    """
    if not sql:
        raise RuntimeError("The sql must be not none!")

    database_config = DatabaseConfig()
    database_config.name = 'mng'
    db = DatabaseWrapper(database_config)
    sqls = [sql] if type(sql) == str else sql
    try:
        db.begin_transaction()
        for index, execute_sql in enumerate(sqls):
            if parameters:
                db.execute(execute_sql, parameters[index])
            else:
                db.execute(execute_sql)
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(f'{error}: {str(e)} ' if error else error)
    finally:
        db.close()


def query_teventmodel_by_id(model_id):
    """
    根据模型编号查询t_event_model表数据。若该模型编号查询出的数据行数不为1则抛出RuntimeError异常。

    :param model_id: string. 模型编号
    :return EventModel.实体对象。
    """
    database_config = DatabaseConfig()
    database_config.name = 'mng'
    db = DatabaseWrapper(database_config)
    try:
        sql = f"SELECT * FROM t_event_model WHERE model_id = %s"
        t_event_model = db.query(sql, (model_id,), result_type=QueryResultType.BEAN, wild_class=EventModel)
        if len(t_event_model) != 1:
            raise RuntimeError(f"Query error, the {model_id} cannot get only one row")
        return t_event_model[0]
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


def __combin_tables_transform_sql(tables: list):

    sql_select_part = ""
    for index, table_name in enumerate(tables):
        sql_select_part = sql_select_part + f"t{index}.*"
        if (index+1) < len(tables):
            sql_select_part = sql_select_part + ","

    sql_from_part = ""
    mark_join = []  # 变量用于存放每张表的表别名，可用于拼接join部分
    for index, table_name in enumerate(tables):
        table_alia = f"t{index}"
        sql_from_part = sql_from_part + f"{table_name} {table_alia}"
        mark_join.append(table_alia)
        if len(mark_join) == 2:
            sql_from_part = sql_from_part + f" ON {mark_join[0]}.rqsj = {mark_join[1]}.rqsj"
            mark_join = [mark_join[0]]
        if (index+1) < len(tables):
            sql_from_part = sql_from_part + " JOIN "

    sql = f"SELECT {sql_select_part} FROM {sql_from_part}"

    return sql


def combine_tables_by_name(tables: list):
    """
    根据传入的数据表列表（数据表名称）拼接成一张表。

    :param tables: array. 数据表名称
    :return dataframe.多个数据表拼接而成的二维表。
    """
    sql = __combin_tables_transform_sql(tables)
    logger.debug(f"Combine tables sql {sql}")

    database_config = DatabaseConfig()
    database_config.name = 'alg'
    db = DatabaseWrapper(database_config)
    try:
        result = db.query(sql, result_type=QueryResultType.PANDAS)
        return result
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()


if __name__ == '__main__':
    tables = ['data_xlshuju_1', 'data_xlshuju_2']
    sql = combine_tables_by_name(tables)
    print(sql)
