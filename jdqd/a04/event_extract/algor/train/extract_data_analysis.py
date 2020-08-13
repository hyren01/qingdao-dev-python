# coding : utf-8
# 解析坐标以及事件的状态
import os
import time
import traceback
from tqdm import tqdm
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path
from jdqd.common.event_emm.data_utils import file_reader, save_json, valid_dir, valid_file


def get_events(ann_content):
    '''
    传入标签字符串，将事件全部解析出来
    :param ann_content: 标注文件
    :return: 事件列表[{"time": , "loc":, "subject": , "trigger": ,"object": , "privative":}, ]
    '''
    all_rows = ann_content.split("\n")
    R = {}
    T = {}
    # R {'T5': ['T7', 'T6'], 'T13': ['T15', 'T16', 'T2']}
    # T {'T5': ['trigger', ["部署", [1,5], happened]],}
    for once in all_rows:
        if once:
            if once.startswith("R"): # 依赖关系
                key = once.split("\t")[1].split(" ")[2].split(":")[1] # 获取动词tag
                value = once.split("\t")[1].split(" ")[1].split(":")[1] # 获取非动词论元tag
                if key in R:
                    R[key].append(value)
                else:
                    R[key] = [value]
            elif once.startswith("T"): # 事件论元
                key = once.split("\t")[0] # 论元tag-->T
                tag = once.split("\t")[1].split(" ")[0].lower() # 论元类型
                id = (once.split("\t")[1].split(" ")[1], once.split("\t")[1].split(" ")[2]) # 论元下标
                value = [once.split("\t")[2], id] # 论元字符
                T[key] = [tag, value]
            elif once.startswith("A"): # 动词状态
                key = once.split("\t")[1].split(" ")[1] # 动词tag
                state = once.split("\t")[1].split(" ")[2].lower() # 动词的状态值
                T[key][1].append(state)

    events = []
    for once in R.items():
        events.append({"time": [], "loc": [], "subject": [], "trigger": [], "object": [], "privative": [], "state": ""})
        if len(T[once[0]][1]) == 3:
            events[-1][T[once[0]][0]].append(T[once[0]][1][0:2]) # 给定动词以及动词下标
            events[-1]["state"] = T[once[0]][1][-1] # 给定动词状态
        else:
            events[-1][T[once[0]][0]].append(T[once[0]][1]) # 状态没有标定就全部给到，状态默认为空

        # 根据补充事件其他论元以及下标
        for i in once[1]:
            events[-1][T[i][0]].append(T[i][1])


    return events


def data_process(file_name):
    '''
    传入单个文件名，输出处理好的事件列表
    :param file_name: 文件名
    :return: 此文件的事件列表
    [{"sentence": , "events" :[{"time": , "loc":, "subject": , "trigger": ,"object": , "privative":}, ]}]
    '''
    # 给定标注文件
    ann_file = f"{file_name}.ann"
    txt_file = f"{file_name}.txt"
    # 获取两个标注文件的内容
    ann_content = file_reader(ann_file)
    txt_content = file_reader(txt_file)
    # 获取ann文件中的事件
    all_events = get_events(ann_content)
    # 获取句子
    sentences = txt_content.split("\n")

    initial_data = {}
    for once in sentences:
        if once:
            for event in all_events:
                if once in initial_data:
                    initial_data[once].append(event)
                else:
                    initial_data[once] = [event]
    data = []
    sentence_num = 0
    event_num = 0
    for once in initial_data:
        data.append({"sentence":once, "events":initial_data[once]})
        event_num += len(initial_data[once])
    sentence_num +=len(data)
    return data , sentence_num, event_num


def execute(raw_dir, target_dir):
    """
    传入原始标注数据文件夹路径和解析后文件存放的路径，按照时间生成json文件名称，将解析好的数据保存到目标文件夹
    :param raw_dir: 存放原始标注数据的文件夹
    :param target_dir: 存放解析后数据的文件夹
    :return: status--解析状态， corpus_num--数据量
    """
    # 存放所有解析后的数据
    all_datas = []
    # 语料中的句子数量
    all_sentence_num = 0
    # 语料中的事件数量
    all_event_num = 0

    try:
        # 判断数据路径是否正确
        if valid_dir(raw_dir):
            # 判断目标文件夹路径是否存在，不存在则创建
            if not valid_dir(target_dir):
                os.makedirs(target_dir)
            file_name = f"{time.strftime('%Y-%m-%d', time.localtime(time.time()))}.json"
            target_file_path = cat_path(target_dir, file_name)
            # 获取文件夹下所有文件的名称
            file_names = os.listdir(raw_dir)
            file_names = list(set(file_name.split(".")[0] for file_name in file_names))
            # 遍历文件进行解析
            for file_name in tqdm(file_names):
                file_path = os.path.join(raw_dir, file_name)
                # 判断两个文件是否都同时存在
                if valid_file(f"{file_path}.ann") and valid_file(f"{file_path}.txt"):

                    # 解析文件获取事件和文件中的句子以及事件数量
                    data, sentence_num, event_num = data_process(file_path)
                    all_datas.extend(data)
                    all_sentence_num +=sentence_num
                    all_event_num +=event_num

            logger.info(f"总共有句子：{all_sentence_num}，总共有事件：{all_event_num}")
            # 将解析后的数据保存到目标文件
            save_json(all_datas, target_file_path)

            return {"status":"success", "results":{"sentences":all_sentence_num, "events":all_event_num}}

        else:
            logger.error(f"存放原始标注数据的文件夹：{raw_dir}没有找到")
            raise FileNotFoundError
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        return {"status":"failed", "results":trace}



if __name__ == "__main__":

    results = execute(raw_dir="D:/work/QingDao_Graph/QingDao_EventExtract/data_util/measure/huangxin_二次矫正", target_dir="D:/work/QingDao_Graph/project")
    print(results)