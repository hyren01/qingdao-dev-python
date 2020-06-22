import numpy as np
from datetime import datetime
from jdqd.a03.event_pred.algor.common import db_conn as db


def obtain_data(data_tables: list):
    """
    从数据表获取数据
      Args:
        data_tables: 使用的数据表名称列表, e.g. [shuju1, shuju2, shuju3]

      Returns:
        数据库查询结果列表, e.g. [rst_shuju1, rst_shuju2, rst_shuju3]
      """
    rsts = []
    for t in data_tables:
        rst = db.query_table(t)
        rst = [list(r) for r in rst]
        rsts.append(rst)
    return rsts


def obtain_events(table_name, event_col, date_col):
    """
    从事件表中获取事件数据
    TODO 注释要更新，43行要确认日期取出来是datetime格式
    Args:
      event_col: 事件在事件表中从0开始的列下标序号
      date_col: 事件对应日期在事件表中从0开始的列下标序号

    Returns:
      事件及其发生日期的二维列表, 第一列为事件, 第二列为日期, datetime.date 类型
    """
    df = db.query_table_2pandas(table_name)
    events = []
    for index, row in df.iterrows():
        # 过滤掉起始日期为空的行
        if row['qssj'] is None:
            continue
        # 过滤掉事件类别字段或日期字段为空的行
        if row[event_col] is None or row[date_col] is None:
            continue
        events.append([row[event_col], row[date_col]])
    #
    # events = [[e[event_col].replace(',', ''), e[date_col]] for e in events if
    #           e[7] is not None]

    # events = [e for e in events_tmp if e[0] is not None and e[1] is not None]
    date_type = type(events[0][1])
    if date_type == str:
        events = [[e[0], datetime(*[int(s) for s in e[1].split('-')])] for e in events if len(e[1]) > 7]

    events.sort(key=lambda x: x[1], reverse=True)
    events = np.array(events, dtype=object)
    events = [[e[0], e[1].date()] for e in events]
    return events


def tidy_table_rst(rst, date_col):
    """
    将从数据库中查询得到的数据整理为模型的输入数据
    Args:
      rst: 从数据表查询得到的数据
      date_col: 日期所在列从0开始的下标序号

    Returns:
      dates: 输入数据的日期列表
      data: 数据表对应的输入数据
    """
    # 最后一列为入库时间
    # TODO 因为表结构问题，现在表中最后一列字段跟特征无关，且是空列，所以写死删除
    rst = [r[:-1] for r in rst]
    rst = np.array(rst)
    # 按照日期列进行排序
    rst = rst[rst[:, date_col].argsort()]
    dates = rst[:, date_col]
    dates = [d.date() for d in dates]
    data = rst[:, 1:]
    # data is None 跟 data == None效果不一样
    data[data == None] = 0
    data = data.astype(int)
    return dates, data


def combine_data(data_tables):
    """
    将多个数据表得到的输入数据合并
    Args:
      data_tables: 使用的数据表名称列表, e.g. [shuju1, shuju2, shuju3]

    Returns:

    """
    """
      :param from_file: 是否从本地文件读取
      :param data_tables: 使用的数据表列表
      :return: 输入数据的日期列表, 合并后的输入数据
      """
    rsts = obtain_data(data_tables)
    data = []
    for r in rsts:
        dates, data_table = tidy_table_rst(r, 0)
        data.append(data_table)
    data = np.concatenate(data, axis=1).astype(np.float)
    zero_var_cols = [i for i in range(data.shape[1]) if len(set(data[:, i])) == 1]
    data = np.array(
      [data[:, i] for i in range(data.shape[1]) if i not in zero_var_cols])
    data = data.T

    return dates, data


def remove_dupli_dates_events(events, event_priority):
    """
    去除事件列表中一天内多次发生的重复事件
    Args:
      events: 事件及其发生日期的二维列表, 第一列为事件, 第二列为事件对应日期列表,
        datetime.date 类型
      event_priority: 同一天多个事件发生时选择保留的事件

    Returns:
      dates_events: 事件时间列表, 日期为 datetime.date 类型
      events: 去重之后的事件列表(不含日期)
    """
    event_dtype = type(events[0][0])

    date_event_dict = {}
    for e, d in events:
        date_event_dict.setdefault(d, []).append(e)

    for d, es in date_event_dict.items():
        if len(es) > 0:
            if event_dtype(event_priority) in es:
                date_event_dict[d] = [event_priority]
            else:
                date_event_dict[d] = es[:1]

    events = list(date_event_dict.items())
    dates_events = [e[0] for e in events]
    events = [e[1][0] for e in events]
    return dates_events, events


def padding_events(dates_x, dates_events, events):
    """
    扩充事件列表, 没发生事件的日期使用0事件表进行填充.
    Args:
      dates_x: 已排序的数据表日期列表, 日期为 datetime.date 类型
      dates_events: 事件表日期列表, 日期为 datetime.date 类型
      events: 事件列表

    Returns: 用 0 填充后的事件列表
    """
    events_p = []
    events_dtype = type(events[0])
    for i, dx in enumerate(dates_x):
        if dx in dates_events:
            events_p.append(events[dates_events.index(dx)])
        else:
            events_p.append(events_dtype(0))
    return np.array(events_p)


def get_events(table_name, dates, event_priority, event_col, date_col):
    """
    获取填充后的事件列表
    # TODO 注释要更新
    Args:
      dates: 已排序的数据表日期列表, 日期为 datetime.date 类型
      event_priority: 同一天多个事件发生时选择保留的事件
      event_col: 事件在事件表中从0开始的列下标序号
      date_col: 事件对应日期在事件表中从0开始的列下标序号

    Returns: 用 0 填充后的事件列表
    """
    events = obtain_events(table_name, event_col, date_col)
    dates_events, events = remove_dupli_dates_events(events, event_priority)
    events_p = padding_events(dates, dates_events, events)
    return events_p
