# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import feedwork.AppinfoConf as appconf
import numpy as np
import pandas as pd
from jdqd.a03.event_pred.algor.common.pgsql_util import query_data_table_2pandas, query_event_table_2pandas
from jdqd.a03.event_pred.algor.common.preprocess import events_one_hot
from jdqd.a03.event_pred.enum.event_type import EventType

# 读取配置文件
__cfg_data = appconf.appinfo["a03"]['data_source']

super_event_type_col = __cfg_data.get('super_event_type_col')   # 大类事件字段名
sub_event_type_col = __cfg_data.get('sub_event_type_col')   # 小类事件字段名
event_table_name = __cfg_data.get('event_table_name')   # 用于训练及预测，在数据库中的数据表名（事件表）

date_col = __cfg_data.get('date_col')      # 时间字段名，仅在事件表中使用
event_priority = __cfg_data['event_priority']

none_event_flag = "0"   # 在有特征数据缺失事件类别的情况下填充的值


def combine_data(data_tables: str):
    """
    获取指定的多张表数据，并且将多张表数据合并一张表，对该表进行按日期列排序、统一日期格式、分离日期列与特征列、
    特征列数据转数值操作。
    Args:
        data_tables: 以','分割的数据表名称列表, e.g. shuju1,shuju2,shuju3
    Returns:
        dates: arrray. 日期字符串数组（数据表中日期列）
        data: dataframe. 数据表（数据表中所有特征列）
    """
    join_data = None
    # 列个数不允许超过1664个，所以把每张表分开查询
    data_tables = data_tables.split(",")
    for table in data_tables:
        pandas_data = query_data_table_2pandas(table)
        # rksj这一列是固定需要删除的空列
        if 'rksj' in list(pandas_data):
            pandas_data = pandas_data.drop(columns=['rksj'])
        if join_data is None:
            join_data = pandas_data
        else:
            join_data = pd.merge(join_data, pandas_data, on="rqsj")

    return __transform_df(join_data, "rqsj")


def get_event(date, event_type):
    """
    根据模型编号获取数据，并对数据进行去重、填充、排序操作。

    :param date: datetime/date.数据表日期
    :param event_type: string.事件类别
    :return events_set: array，事件类别列表
            events_p_oh: array, one-hot 形式的补全事件列表
    """
    event_col = super_event_type_col if event_type == EventType.SUPER_EVENT_TYPE.value else sub_event_type_col
    # 对数据进行去重、填充操作，返回补0后的事件列表。将[['2020-06-24', '11209'], ['2020-06-24', '11011'], ['2020-06-24', '']]
    # 转换成['11209', '0', '11011']。event_priority这是跟客户确认在一天时间内多个事件情况下优先选择的事件。
    events_p = __transform_events(event_table_name, date, event_priority, event_col, date_col)
    # 对事件进行去重且排序
    events_set = sorted(set(events_p))  # 事件类别集合
    # one-hot处理指的是，使用重且排序的事件类型列表（如：[0,1,2]），与原始事件类型列表数据（如：[1,2,0,1,1]）进行数据转换操作，
    # 转换后的结果是[[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0]]，这是模型使用时需要的数据
    events_p_oh = events_one_hot(events_p, events_set)  # 补 0 后 one-hot 形式的事件列表

    return events_set, events_p_oh


def __transform_df(df, date_col):
    """
    对传入的dataframe进行统一日期格式、分离日期列与特征列、特征列数据转数值操作。该方法会把date_col参数在dataframe
    中删除，并且单独提取出date_col指定的列作为返回值。
    Args:
        df: dataframe. 数据库中的表
        date_col: string. 表中的时间字段名
    Returns:
        dates: arrray. 日期字符串数组
        data: dataframe. 数据表
    """

    # 删除按照日期列进行排序
    # df = df.sort_values(by=date_col)
    # 单独提取出日期整列，得到Series对象
    dates = df[date_col].astype('str')
    # 得到array对象
    dates = [__transform_date_str(date_str) for date_str in dates]
    # 提取出除日期列外的所有列数据，得到DataFrame对象
    data = df.drop(columns=[date_col])
    data = data.where(~data.isna(), other=0).astype(int)    # 将特征数据中为none的数据替换为0，并将所有特征转换为数值类型

    return dates, data


def __transform_date_str(date_str):
    if date_str is None or date_str == '':
        return
    date_str = date_str.split(" ")[0]  # 处理yyyy-MM-dd HH:mm:ss的情况，只要yyyy-MM-dd
    # 尽可能将各种格式的日期格式转换为统一的yyyy-MM-dd格式，日期数据存在yyyy-M的情况
    date_str = date_str.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").strip()
    # pattern = re.compile(r'\d+')
    # date_part = re.findall(pattern, date_str)
    # date_part_len = len(date_part)
    # yyyy = date_part[0] if date_part_len > 0 else ''
    # # 若月份与日期字符长度为1则前补0，否则原样输出
    # mm = (f'0{date_part[1]}'if len(date_part[1]) == 1 else date_part[1]) if date_part_len > 1 else ''
    # dd = (f'0{date_part[2]}'if len(date_part[2]) == 1 else date_part[2]) if date_part_len == 3 else ''

    return date_str


def __transform_events(table_name, dates, event_priority, event_col, date_col):
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
    # events = __obtain_events(table_name, event_col, date_col)
    events = query_event_table_2pandas(table_name, event_col, date_col)
    event_dates, events = __get_dates_and_events(events, event_priority, event_col, date_col)
    normal_events = __padding_events(dates, event_dates, events)

    return normal_events


def __get_dates_and_events(events_df, event_priority, event_col, date_col):
    """
    该方法会去除事件类别列表中一天内多次发生的重复事件、统一事件日期字符串格式、分离日期列与事件列。
    Args:
      events_df: dataframe.事件表数据,第一列为事件类别，第二列为事件日期
      event_priority: 同一天多个事件发生时选择保留的事件
      event_col: string，事件类别字段名
      date_col: string，事件日期字段名

    Returns:
      dates: array，事件时间列表
      events: array，去除重复事件的事件类别列表(不含日期)
    """

    date_event_dict = {}
    # 对事件表中的数据按日期归并，即key是日期，value是事件表中去除日期的事件类别列表（事件号）
    for _, row in events_df.iterrows():
        date_event_dict.setdefault(__transform_date_str(row[date_col]), []).append(row[event_col])
    # 如果归并的数据中（events变量）的事件类别列表有多个，若event_priority（指定事件）存在则只保留该事件，
    # 否则取事件类别列表中的第一个事件
    event_dtype = type(events_df.at[0, event_col])     # 取第一行event_col字段的数据，获得其数据类型
    for date, events in date_event_dict.items():
        # 若同一天内发生了多个事件，优先填入event_priority，否则只取第一个
        if len(events) > 1:
            if event_dtype(event_priority) in events:
                date_event_dict[date] = event_dtype(event_priority)
            else:
                date_event_dict[date] = events[0]
        else:
            # len(events)一定大于0，因为events在查询数据库时过滤了空值
            date_event_dict[date] = events[0]
    # date_event_dict.items()返回的是tuple，即：(key, value)，key为事件日期字符串、value为事件类别字符串
    events = list(date_event_dict.items())
    dates = [event[0] for event in events]   # 单独提取出事件日期
    events = [event[1] for event in events]  # 单独提取出事件列表

    return dates, events


def __padding_events(data_dates, event_dates, events):
    """
    扩充事件类别列表, 没发生事件的日期使用0事件表进行填充.
    Args:
      data_dates: 数据表事件日期列表
      event_dates: 事件表事件日期列表
      events: 事件表事件类别列表

    Returns: 用 0 填充后的事件列表
    """
    events_p = []
    # 客户生产机跟测试机上的数据类型不一样，这里做统一数据类型操作，统一为字符型
    for index, date in enumerate(data_dates):
        # 若数据表中事件日期与事件表中的事件日期一致，则根据事件表的事件日期下标找到对应的事件类别，
        # 并按照数据表顺序记录事件类别，否则认为该数据特征下没有发生事件，事件类别填充为0
        if date in event_dates:
            events_p.append(str(events[event_dates.index(date)]))
        else:
            events_p.append(none_event_flag)
    return np.array(events_p)
