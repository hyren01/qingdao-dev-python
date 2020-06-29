# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np
from datetime import datetime
from jdqd.a03.event_pred.algor.common.pgsql_util import query_event_table_2pandas, query_data_table_2pandas


def obtain_events(table_name, event_col, date_col):
    """
    从事件表中获取事件数据，并将数据转换为二维表。
    Args:
        table_name: 数据库中的数据表名（事件表）
        event_col: 事件在事件表中从0开始的列下标序号
        date_col: 事件对应日期在事件表中从0开始的列下标序号

    Returns:
        事件及其发生日期的二维列表, 第一列为事件, 第二列为日期, datetime.date 类型
    """
    df = query_event_table_2pandas(table_name)
    events = []
    for index, row in df.iterrows():
        # 过滤掉起始日期为空的行
        if row['qssj'] is None:
            continue
        # 过滤掉事件类别字段或日期字段为空的行
        if row[event_col] is None or row[date_col] is None:
            continue
        events.append([row[event_col], row[date_col]])

    date_type = type(events[0][1])  # 随便取一条数据出来获取日期类型
    if date_type == str:    # 如果日期数据为字符类型，则遍历每一行数据并转换日期数据为日期类型
        events = [[event[0], datetime(*[int(s) for s in event[1].split('-')])] for event in events if len(event[1]) > 7]

    events.sort(key=lambda x: x[1], reverse=True)
    events = np.array(events, dtype=object)
    events = [[e[0], e[1].date()] for e in events]
    return events


# def tidy_table_rst(rst, date_col):
#     """
#     将从数据库中查询得到的数据整理为模型的输入数据。
#     # TODO date_col是写死的字段下标，要改掉
#     Args:
#         rst: 从数据表查询得到的数据
#         date_col: 日期所在列从0开始的下标序号
#
#     Returns:
#         dates: array，输入数据的日期列表
#         data: dataframe，数据表对应的输入数据
#     """
#     # TODO 因为表结构问题，现在表中最后一列字段跟特征无关，且是空列，所以写死删除
#     rst = [r[:-1] for r in rst]     # 最后一列为入库时间，跟特征无关，且是空列，所以写死删除
#     rst = np.array(rst)
#     rst = rst[rst[:, date_col].argsort()]   # 按照日期列进行排序，写死日期所在列下标为0
#     dates = rst[:, date_col]     # 单独提取出日期数据
#     # TODO 这里要求从数据库中读出来的日期数据必须是date/datetime类型，SQL中转成字符型，程序再操作
#     dates = [d.date() for d in dates]
#     data = rst[:, 1:]   # 提取出除日期之外的数据作为特征数据
#     # data is None 跟 data == None效果不一样
#     data[data == None] = 0  # 将特征数据中为none的数据替换为0，TODO 要确认真实数据中是否有none的情况，这里先留着
#     data = data.astype(int)
#     return dates, data


# def combine_data(data_tables):
#     """
#     将多张指定数据表中的数据合并为一张数据表
#     Args:
#         data_tables: 指定的数据表名称列表，如：[shuju1, shuju2, shuju3]
#
#     Returns:
#             date: string，日期
#             data: array，事件类别列表
#     """
#
#     # TODO 使用SQL拼接数据，只需要单独提取出date即可，返回pandas，顺带处理用下标去除空列的问题
#
#     rsts = obtain_data(data_tables)     # 存放的是多张表数据的数组
#     data = []
#     for table_data in rsts:
#         # TODO 这个有时间要改掉，日期只分析一次即可，还需要讨论
#         date, data_table = tidy_table_rst(table_data, 0)   # 在一个批次内，所有表的日期一样
#         data.append(data_table)
#     # 将多张表数据通过第一列进行合并为一张表数据
#     data = np.concatenate(data, axis=1).astype(np.float)
#     # 没一列数据取值并进行set（去重）操作，判断每一列是否只有一个原始，去除只有一个特征的列
#     # zero_var_cols = [i for i in range(data.shape[1]) if len(set(data[:, i])) == 1]
#     # data = np.array([data[:, i] for i in range(data.shape[1]) if i not in zero_var_cols])
#     # data = data.T
#
#     return date, data


def remove_dupli_dates_events(events, event_priority):
    """
    去除事件类别列表中一天内多次发生的重复事件
    Args:
      events: 事件及其发生日期的二维列表, 第一列为事件类别, 第二列为事件对应日期,
      datetime.date 类型
      event_priority: 同一天多个事件发生时选择保留的事件

    Returns:
      dates_events: array，事件时间列表, 日期为 datetime.date 类型
      events: array，去重之后的事件列表(不含日期)
    """
    event_dtype = type(events[0][0])

    date_event_dict = {}
    # 对事件表中的数据按日期归并，即key是日期，value是事件表中去除日期的事件类别列表（事件号）
    for event, date in events:
        date_event_dict.setdefault(date, []).append(event)
    # 如果归并的数据中（events变量）的事件类别列表有多个，若event_priority（指定事件）存在则只保留该事件，
    # 否则取事件类别列表中的第一个事件
    for date, events in date_event_dict.items():
        if len(events) > 0:
            # 使event_priority的数据类型与事件类别保持一致
            if event_dtype(event_priority) in events:
                date_event_dict[date] = [event_dtype(event_priority)]
            else:
                date_event_dict[date] = events[:1]
    # date_event_dict.items()返回的是tuple，即：(key, value)，key为日期、value为事件类别
    events = list(date_event_dict.items())
    dates_events = [event[0] for event in events]   # 单独提取出事件日期
    events = [event[1][0] for event in events]  # 单独提取出事件列表
    return dates_events, events


def padding_events(dates_x, dates_events, events):
    """
    扩充事件类别列表, 没发生事件的日期使用0事件表进行填充.
    Args:
      dates_x: 已排序的数据表日期列表, 日期为 datetime.date 类型
      dates_events: 事件表日期列表, 日期为 datetime.date 类型
      events: 事件列表

    Returns: 用 0 填充后的事件列表
    """
    events_p = []
    # 客户生产机跟测试机上的数据类型不一样，这里做统一数据类型操作，统一为字符型
    for index, date in enumerate(dates_x):
        if date in dates_events:
            events_p.append(str(events[dates_events.index(date)]))
        else:
            events_p.append('0')
    return np.array(events_p)


def get_events(table_name, dates, event_priority, event_col, date_col):
    """
    获取事件列表，该方法会对指定数据表中的数据进行去重、填充操作。
    Args:
      table_name: string，数据库表名
      dates: array，已排序的数据表日期列表, 日期为 datetime.date 类型
      event_priority: string，同一天多个事件发生时选择保留的事件
      event_col: string，事件类别字段名
      date_col: string，事件日期字段名

    Returns: array，事件类别列表
    """
    events = obtain_events(table_name, event_col, date_col)
    dates_events, events = remove_dupli_dates_events(events, event_priority)
    events_p = padding_events(dates, dates_events, events)
    return events_p
