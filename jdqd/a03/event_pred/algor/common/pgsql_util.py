# -*- coding: utf-8 -*-
import time
import numpy as np
from feedwork.utils import logger, UuidHelper
from jdqd.a03.event_pred.algor.common import preprocess as pp, db_conn


def update_model_status(model_id, status="3"):
    """训练模型错误时, 将 t_event_model 表 status 字段更新为 3
    """
    sql = "UPDATE t_event_model SET status = %s WHERE model_id = %s"
    db_conn.modify(sql, [(status, model_id)])


def model_train_done(model_id, file_path_list):
    sqls = []
    params = []
    for model_fp in file_path_list:
        sql = "INSERT INTO t_event_model_file(file_id, file_url, model_id) VALUES(%s, %s, %s) "
        sqls.append(sql)
        uuid = UuidHelper.guid()
        param = (uuid, model_fp, model_id)
        params.append(param)
    db_conn.modify(sqls, params)


def model_eval_done(model_id, date_str, time_str, status="3"):
    sql = f"UPDATE t_event_model SET status = %s, tran_finish_date = %s, tran_finish_time = %s WHERE model_id = %s"
    param = [(status, date_str, time_str, model_id)]
    db_conn.modify(sql, param)


def insert_into_model_detail(sub_model_names, model_id):
    """将子模型信息入库
    Args:
      sub_model_names: 子模型名称, 形如 model_name-input_len-output_len-n_pca
      model_id: 总模型的 id
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
    db_conn.modify(sqls, params)

    return detail_ids


def insert_into_model_train(detail_ids, outputs_list, events_set, status):
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
    db_conn.modify(sqls, params)


def insert_model_test(event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
                      status, detail_id):
    test_id = UuidHelper.guid()
    sql = "insert into t_event_model_test(test_id, event_name, num, false_rate, recall_rate, false_alarm_rate, " \
          "tier_precision, tier_recall, bleu, status, detail_id)values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    param = [(test_id, event, event_num, false_rate, recall_rate, false_alarm_rate, tier_precision, tier_recall, bleu,
              status, detail_id)]
    error = '插入子模型评估结果失败'
    db_conn.modify(sql, param, error)


def insert_model_tot(scores, events_num):
    """
      将模型评估结果插入数据库
      :param scores:
      :param events_num:
      :return:
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
    db_conn.modify(sqls, params, '存入top模型出错')


def query_sub_models_by_model_id(model_id):
    sql = "SELECT detail_id, score FROM t_event_model_tot WHERE detail_id IN " \
          "(SELECT detail_id FROM t_event_model_detail WHERE model_id = %s) "
    results1 = db_conn.query(sql, 'mng', (model_id,))
    detail_ids = [r[0] for r in results1]
    detail_ids = tuple(detail_ids)
    sql2 = "SELECT model_name, detail_id FROM t_event_model_detail WHERE detail_id IN %s"
    results2 = db_conn.query(sql2, 'mng', (detail_ids,))

    return results2


def query_event_type_by_id(model_id):
    sql = "SELECT event_type FROM t_event_model WHERE model_id = %s"
    result = db_conn.query(sql, 'mng', (model_id,))
    if len(result) != 1:
        raise RuntimeError(f"Query error, model_id cannot get a row {model_id}")
    # db_conn.query返回的是[['xxx']]
    return result[0][0]


def query_sub_models_by_model_name(model_name):
    sql = "SELECT detail_id, score FROM t_event_model_tot WHERE detail_id IN " \
          "(SELECT detail_id FROM t_event_model_detail WHERE model_name like %s) "
    results1 = db_conn.query(sql, 'mng', (f'{model_name}%',))
    detail_ids = [r[0] for r in results1]
    detail_ids = tuple(detail_ids)
    sql2 = "SELECT model_name, detail_id FROM t_event_model_detail WHERE detail_id IN %s"
    results2 = db_conn.query(sql2, 'mng', (detail_ids,))

    return results2


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
    results = db_conn.query(sql, 'mng', param)
    detail_id_dates = {}
    for r in results:
        detail_id_dates.setdefault(r[0], set()).add(r[1])

    return detail_id_dates


def insert_new_pred_task(task_id, model_id, model_name, tables_name, epoch, time_str):
    sql = "insert into t_event_task(task_id, model_id, model_name, tables_name, epoch, create_time) values " \
          "(%s, %s, %s, %s, %s, %s)"
    param = (task_id, model_id, model_name, tables_name, str(epoch), time_str)
    db_conn.modify(sql, 'mng', [param])


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
        param = (detail_id, str(d))
        params.append(param)
        sqls.append(sql)
    db_conn.modify(sqls, 'mng', [params])


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
        db_conn.modify(sqls, params, error)

        error_his = f't_event_task_rs_his表预测结果插入出错'
        db_conn.modify(sqls_hist, params_hist, error_his)


def update_task_status(task_id, status="3"):
    sql = f"UPDATE t_event_task SET status = %s WHERE task_id = %s"
    param = (status, task_id)
    db_conn.modify(sql, [param])


def predict_task_done(task_id, date_str, time_str, date_data_pred, status="3"):
    sql = "UPDATE t_event_task SET status = %s, task_finish_date = %s, " \
          "predict_end_date = %s, task_finish_time = %s WHERE task_id = %s"
    param = (status, date_str, str(date_data_pred), time_str, task_id)
    db_conn.modify(sql, [param])
