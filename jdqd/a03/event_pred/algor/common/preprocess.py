# coding=utf-8
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from datetime import date
from feedwork.utils.FileHelper import cat_path


def one_hot_dict(events_set):
    """
    根据事件列表生成每个事件类型与对应one-hot形式向量的字典
      Args:
        events_set: 事件集合

      Returns:
        每个事件类型与对应one-hot形式向量的字典. e.g. {event1: [0, 1, 0, 0, 0]}
      """
    m = len(events_set)
    eye = np.eye(m)
    oh_dict = {}
    for i, r in zip(events_set, eye):
        oh_dict[i] = r
    return oh_dict


def events_one_hot(events_p, events_set):
    """
    将事件列表转化成one-hot形式表示的矩阵。one-hot处理指的是，使用重且排序的事件类型列表（如：[0,1,2]），
    与原始事件类型列表数据（如：[1,2,0,1,1]）进行数据转换操作，转换后的结果是[[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,1,0]]，
    这是模型使用时需要的数据
      Args:
        events_p: 无事件用 0 补全后的事件列表
        events_set: 去重排序后的事件集合

      Returns:
        one-hot 形式的补全事件列表
      """
    oh_dict = one_hot_dict(events_set)
    events_p_oh = np.array([oh_dict[a] for a in events_p])
    return events_p_oh


def events_binary(events_p_oh, event_col):
    """
    将one-hot形式的事件矩阵转换成使用[0, 1]表示的某个特定事件是否发生的列表
    Args:
      events_p_oh: one-hot 形式的补全事件列表
      event_col: 需转换的事件在事件列表中对应的列数

    Returns:
      使用(0, 1)表示的某个特定事件是否发生的列表
    """
    return np.equal(events_p_oh[:, event_col], 1).astype(int)


def get_recur_events_rows(events_p_oh):
    """
    获取多次出现的事件与其对应发生的日期在输入数据中的行数列表的字典
    Args:
      events_p_oh: one-hot 形式的补全事件列表

    Returns:
      多次出现的事件与其对应发生的日期在输入数据中的行数列表的字典, e.g. {11209: [3, 456, 567]}

    """
    recur_events_cols = np.where(np.sum(events_p_oh[..., 1:], axis=0) > 1)[0] + 1
    recur_events_rows = {c: np.where(events_p_oh[:, c] == 1)[0] for c in recur_events_cols}
    return recur_events_rows


def apply_pca(pca_n, pca_dir, data, reload=False):
    """
    对输入数据进行pca降维
    Args:
      pca_n: 降维后维度
      pca_dir: pca模型所在目录
      data: 数据表数据
      reload: 是否加载已存在的pca模型文件

    Returns:
      降维操作后的数据
    """
    pca_fp = cat_path(pca_dir, f"{pca_n}fit_pca")
    if reload:
        data_pca = joblib.load(pca_fp).transform(data)
    else:
        pca = PCA(n_components=pca_n)
        data_pca = pca.fit_transform(data)
        joblib.dump(pca, pca_fp)
    return data_pca


def parse_date_str(date_str):
    """
    将 yyyy-mm-dd 格式的日期字符串转换成 datetime.date 类型
    """
    date_ = date_str.split('-')
    date_ = [int(d) for d in date_]
    date_ = date(*date_)
    return date_


def date_to_index(date_str, dates):
    """
    计算日期字符串在数据集中对应的下标
    Args:
      date_str: yyyy-mm-dd 格式的日期字符串
      dates: 数据表日期列表

    Returns:
      日期在数据表日期列表中对应的下标
    """
    # date_ = parse_date_str(date_str)
    if date_str not in dates:
        raise RuntimeError(f"预测开始日期不在样本数据日期范围内 {date_str}")
    return dates.index(date_str)


def gen_input_by_input_start_row(values_pca, input_start_row, input_len):
    """
    根据 encoder 输入序列开始下标以及 encoder 输入长度生成一个输入样本
    Args:
      values_pca: 降维后数据表数据
      input_start_row: encoder 输入序列开始下标
      input_len: encoder 输入序列长度

    Returns:
      一个 encoder 输入序列样本, 为降维后数据表数据的一个切片
    """
    input_sample = values_pca[input_start_row: input_start_row + input_len]
    return input_sample


def gen_output_by_input_start_row(events_p_oh, input_start_row, input_len, output_len):
    """
    根据 encoder 输入序列开始下标生成一个decoder输出样本
    Args:
      events_p_oh: one-hot 形式的补全事件列表
      input_start_row: encoder 输入序列开始下标
      input_len: encoder 输入序列长度
      output_len: decoder 输出序列长度, 即预测天数

    Returns:
      decoder 输出序列, 为事件列表的一个切片

    """
    output_sample = events_p_oh[input_start_row + input_len: input_start_row + input_len + output_len]
    return output_sample


def flatten_outputs(outputs):
    """
    将 2d decoder 输出列表转为 1d. 取输出序列的第一天元素作为 1d 列表的元素, 保留最后一个
    输出样本的所有元素.
    Args:
      outputs: 2d decoder 输出列表. e.g. [[0, 11209, 0], [0, 0, 0], [11209, 0, 11011]]

    Returns:
      选取 2d 列表第一列组成的新列表, 2d 列表最后一个元素保留. e.g. [0, 0, 11209, 0, 11011]
    """
    outputs_flatten = [o[0] for o in outputs]
    outputs_flatten.extend(outputs[-1][1:])
    return np.array(outputs_flatten)


def get_event_num(outputs, events_set):
    """
    获取一段时间内出现的各个事件的数量
    Args:
      outputs: decoder 输出事件列表
      events_set: 有序事件集合

    Returns:
      decoder 输出事件列表里出现的事件与其出现次数的字典
    """
    outputs_flatten = flatten_outputs(outputs)
    events_num = {}
    for i, e in enumerate(events_set):
        n_event = int(np.sum(outputs_flatten[:, i]))
        events_num[e] = n_event
    return events_num


def gen_samples_by_pred_date(values_pca, events_p_oh, input_len, output_len, dates, pred_start_date, pred_end_date):
    """
    以下所有范围中的滑动，步长都是1，每次滑动得到一个样本，一个样本包括特征跟事件类别。该方法返回样本数组。
    1、min_output_start_index、max_output_end_index日期下标范围内值的是事件数据的数据范围，
        事件在该范围内每次根据滑动窗口（预测天数）滑动，最后日期下标不会超过max_output_end_index；
    2、特征数据在min_output_start_index、max_output_end_index基础上偏移向前偏移input_len天，得到特征数据范围；
    3、max_output_end_index - dates + 1能得到事件最大滑动范围，事件是根据预测天数（dates）进行滑动；
    4、max_input_end_row - input_len + 1能得到特征最大滑动范围，min_output_start_index - input_len
        能得到特征开始滑动日期下标，特征是根据滞后期滑动；
    5、模型需要的数据是[[1],...,[15],[16],...,[20]]，[1]到[15]是特征数据，[16]到[20]是事件类别数据，
        含义是：根据[1]到[15]范围内15天的特征，预测未来5天（[16]到[20]确认）的事件类别。
    Args:
      values_pca: 降维后数据表数据
      events_p_oh: one-hot 形式的补全事件列表
      input_len: encoder 输入序列长度
      output_len: decoder 输出序列长度, 即预测天数
      dates: 数据表日期列表
      pred_start_date: 开始日期
      pred_end_date: 截止日期

    Returns:
      decoder 输出序列在开始预测日期与预测截止日期之间的样本, 包含输入及输出序列
    """
    # 基于pred_start_date、pred_end_date在dates中搜索，能知道当前日期所在样本数据中的位置
    min_output_start_index = date_to_index(pred_start_date, dates)
    if min_output_start_index is None:
        raise RuntimeError("输入的开始日期无法在数据中找到")
    max_output_end_index = date_to_index(pred_end_date, dates)
    if min_output_start_index is None:
        raise RuntimeError("输入的结束日期无法在数据中找到")
    # 该行代码能找到特征数据开始日期下标 TODO 该行代码的加减法由日期排序方式来决定，日期升序排序即为减法
    min_input_start_row = min_output_start_index - input_len
    if min_input_start_row < 0:
        raise RuntimeError("输入的开始日期已经超过数据的最小日期范围")
    # min_input_start_row = max(min_input_start_row, 0)
    # 该行代码能找到特征数据结束日期下标 TODO The same
    max_input_end_row = max_output_end_index - output_len
    if min_input_start_row < 0:
        raise RuntimeError("输入的结束日期过小，无法满足前推预测天数")
    # 该行代码找到特征数据中最大滑动日期下标
    max_input_start_row = max_input_end_row - input_len + 1
    if min_input_start_row < 0:
        raise RuntimeError("输入的结束日期过小，无法满足前推滞后期天数")
    # 生成特征样本数据
    inputs = [gen_input_by_input_start_row(values_pca, start_row, input_len)
              for start_row in range(min_input_start_row, max_input_start_row + 1)]
    # 生成事件样本数据
    outputs = [gen_output_by_input_start_row(events_p_oh, start_row, input_len, output_len)
               for start_row in range(min_input_start_row, max_input_start_row + 1)]

    return np.array(inputs), np.array(outputs)


def gen_inputs_by_pred_start_date(values_pca, input_len, dates, pred_start_date):
    """
    获取起始预测日期之后的 encoder 输入数据
    Args:
      values_pca: 降维后数据表数据
      input_len: encoder 输入序列长度
      dates: 数据表日期列表
      pred_start_date: 开始预测日期

    Returns:
      起始预测日期之后对应的 encoder 输入序列, 对应的日期列表
    """
    pred_start_row = date_to_index(pred_start_date, dates)
    # 用于预测的每一个样本对应的特征数据开始下标
    min_input_start_row = pred_start_row - input_len
    if min_input_start_row < 0:
        raise RuntimeError("开始预测日期错误，该日期之前已无数据")
    # 特征表样本数据行数（日期序列） - 滞后期（input_len）能得到预测的每一个样本对应的特征数据结束下标
    max_input_start_row = len(dates) - input_len
    input_ = [gen_input_by_input_start_row(values_pca, data_index, input_len)
              for data_index in range(min_input_start_row, max_input_start_row + 1)]
    dates_ = dates[pred_start_row:]
    return np.array(input_), dates_
