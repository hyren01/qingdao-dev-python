import pandas as pd
import datetime
import jdqd.a03.event_pred.restapp.main as main_
from jdqd.a03.event_pred.algor.train import model_evalution as meval
from jdqd.a03.event_pred.algor.common import pgsql_util as pgsql, preprocess as pp, obtain_data as od

'''
使用原始固定数据集作为输入，调用预测程序并使用固定的评估逻辑，对预测结果做评判。
'''

#同一天发生多个事件时选择保留的事件
event_priority = 11209
#事件在事件表中从0开始的列下标序号
event_col = 1
#事件对应日期在事件表中从0开始的列下标序号
date_col = 3
#模型名称前缀
model_name = 'test_suit_model'
#测试数据路径
data_path = 'true_data_dalei.csv'
#训练数据起始日期
train_start_date = '2016-01-01'
#训练数据结束日期
train_end_date = '2016-08-24'
#测试数据起始日期
start_date = '2016-08-25'
#测试数据结束日期
end_date = '2019-03-15'
#预测天数
output_len = 5
#最小降维度
min_dim = 5
#最大降维度
max_dim = 61
#最小滞后期
min_input_len = 5
#最大滞后期
max_input_len = 51
#循环步长
step = 5
#网络单元
num_units = 128
#训练每批次个数
batch = 64
#训练批次
epoch = 100
#一次训练任务对应总模型id
model_id = '1'

def load_data(data_path):
    '''
    读取测试数据
    param data_path: 测试数据路径
    return 测试数据、对应日期
    '''
    dataset = pd.read_csv(data_path, header=None)
    dates = dataset.iloc[:, 0].values
    dates = [datetime.date(*[int(s) for s in d.split('-')]) for d in dates]
    data = dataset.iloc[:, 1:].values
    return dates, data

dates, data = load_data(data_path)
#data数据对应的事件列表
events_p = data[:, -1]
#事件列表对应事件库                 
events_set = pp.get_events_set(events_p)
#事件对应one_hot
events_p_oh = pp.events_one_hot(events_p, events_set)
#事件个数
n_classes = len(events_set)

def model2score():
    '''
    根据设置参数循环遍历并训练模型，得到各模型得分
    return scores(list)
    '''
    sub_model_dirs, sub_model_names, outputs_list, params_list = main_.train_over_hyperparameters(
            model_name,
            data,
            events_p_oh,
            dates,
            train_start_date,
            train_end_date,
            output_len,
            min_dim,
            max_dim,
            min_input_len,
            max_input_len,
            step,
            num_units,
            batch,
            epoch)
    detail_ids = pgsql.insert_into_model_detail(sub_model_names, model_id)

    scores, events_num = meval.evaluate_sub_models(data, dates, detail_ids,
                                                   sub_model_dirs,
                                                   params_list, events_p_oh,
                                                   events_set, n_classes,
                                                   start_date,
                                                   end_date)
    return scores
    
scores = model2score()
#判断是否优化
for score in scores:
        assert score[0] >= 0.7