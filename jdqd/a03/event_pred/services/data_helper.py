# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import feedwork.AppinfoConf as appconf
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, preprocess as pp, obtain_data as od
from jdqd.a03.event_pred.enum.event_type import EventType

# 读取配置文件
__cfg_data = appconf.appinfo["a03"]['data_source']

super_event_type_col = __cfg_data.get('super_event_type_col')   # 大类事件字段名
sub_event_type_col = __cfg_data.get('sub_event_type_col')   # 小类事件字段名
event_table_name = __cfg_data.get('event_table_name')   # 用于训练及预测，在数据库中的数据表名（事件表）

date_col = __cfg_data.get('date_col')      # 时间字段名
event_priority = __cfg_data['event_priority']


def combine_data(data_tables: list):
    """
    获取指定的多张表数据，并且将多张表数据合并为一个数组。
    Args:
        data_tables: 使用的数据表名称列表, e.g. [shuju1, shuju2, shuju3]
    Returns:
        dataframe.多个表合并成一个dataframe
    """
    join_data = None
    # 列个数不允许超过1664个，所以把每张表分开查询
    for table in data_tables:
        pandas_data = pgsql.query_data_table_2pandas(table, date_col)
        # rksj这一列是固定需要删除的空列
        if 'rksj' in list(pandas_data):
            pandas_data = pandas_data.drop(columns=['rksj'])
        if pandas_data is None:
            join_data = pandas_data
        else:
            join_data = join_data.join(pandas_data, on="rqsj", how="inner")

    return __transform_df(join_data, date_col)


def __transform_df(df, date_col):

    def transform_date_str(date_str):
        if date_str is None or date_str == '':
            return
        date_str = date_str.split(" ")[0]  # 处理yyyy-MM-dd HH:mm:ss的情况，只要yyyy-MM-dd
        # 尽可能将各种格式的日期格式转换为统一的yyyy-MM-dd格式
        date_str = date_str.replace("年", "-").replace("月", "-").replace("日", " ").replace("/", "-").strip()

        return date_str
        # date_split = date_str.split("-")
        # # 开始处理2009-04-05、2015-7-、2014-的问题
        # date_array = [0] * 3     # 初始化一个固定长度为3，且每个元素初始为0的数组
        # for index, date_part in enumerate(date_split):
        #     if date_part.strip() == '':
        #         date_part = 0
        #     date_array[index] = int(date_part)
        #
        # return date(date_array[0], date_array[1], date_array[2])

    # 删除按照日期列进行排序
    df = df.sort_values(by=date_col)
    # 单独提取出日期整列，得到Series对象
    dates = df[date_col]
    # 得到array对象
    dates = [transform_date_str(date_str) for date_str in dates]
    # 提取出除日期列外的所有列数据，得到DataFrame对象
    data = df.drop(columns=[date_col])
    data[data == None] = 0  # 将特征数据中为none的数据替换为0，
    data = data.astype(int)

    return dates, data


def transform_data(date, model_id):
    """
    根据模型编号获取数据，并对数据进行去重、填充、排序操作。

    :param date: datetime/date.数据表日期
    :param model_id: string.模型编号
    :return events_set: array，事件类别列表
            events_p_oh: array, one-hot 形式的补全事件列表
    """
    event_type = pgsql.query_event_type_by_id(model_id)
    event_col = super_event_type_col if event_type == EventType.SUB_EVENT_TYPE.value else sub_event_type_col
    # 对数据进行去重、填充操作，返回补0后的事件列表。将[['2020-06-24', '11209'], ['2020-06-24', '11011'], ['2020-06-24', '']]
    # 转换成['11209', '0', '11011']。event_priority这是跟客户确认在一天时间内多个事件情况下优先选择的事件。
    events_p = od.get_events(event_table_name, date, event_priority, event_col=event_col, date_col=date_col)
    # 对事件进行去重且排序
    events_set = pp.get_events_set(events_p)  # 事件集
    # one-hot处理指的是，使用重且排序的事件类型列表（如：[0,1,2]），与原始事件类型列表数据（如：[1,2,0,1,1]）进行数据转换操作，
    # 转换后的结果是[[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0]]，这是模型使用时需要的数据
    events_p_oh = pp.events_one_hot(events_p, events_set)  # 补 0 后 one-hot 形式的事件列表

    return events_set, events_p_oh
