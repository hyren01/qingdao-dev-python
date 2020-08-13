#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
事件抽取模块的接口，提供事件抽取接口和指代消解接口
"""
import os
import copy
import traceback
import threading
from queue import Queue
from flask_cors import CORS
from flask import Flask, request, jsonify
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path # 模型重载使用
# from jdqd.common.event_emm.data_utils import data_process, get_sentences  # 数据清洗和分句，后期调试可能会用到
from jdqd.a04.event_extract.config import PredictConfig  # 模型重载使用
from jdqd.a04.event_extract.algor.predict.load_all_model import get_state_model, get_extract_model, get_ners_lac, \
    extract_items, get_cameo_model, get_event_cameo

# 加载模型参数
state_model = get_state_model()
trigger_model, object_model, subject_model, loc_model, time_model, negative_model = get_extract_model()
cameo_model = get_cameo_model()

app = Flask(__name__)
CORS(app)
# 事件抽取主队列
EXTRACT_QUEUE = Queue(maxsize=5)
# 指代消解主队列
COREF_QUEUE = Queue(maxsize=5)


@app.route('/reload_model', methods=['GET', 'POST'])
def reload_model():
    """
    从前端接收模型路径，重新加载模型
    :return: status 加载状态
    """
    # 模型路径字典
    model_path = {"extract": "", "cameo": "", "state": ""}

    if request.method == "POST":
        model_path["extract"] = request.form.get("extract", type=str, default=None)
        model_path["cameo"] = request.form.get("cameo", type=str, default=None)
        model_path["state"] = request.form.get("state", type=str, default=None)
        model_version = request.form.get("version", type=str, default=None)
        model_id = request.form.get("model_id", type=str, default=None)
    else:
        model_path["extract"] = request.args.get("extract", type=str, default=None)
        model_path["cameo"] = request.args.get("cameo", type=str, default=None)
        model_path["state"] = request.args.get("state", type=str, default=None)
        model_version = request.args.get("version", type=str, default=None)
        model_id = request.args.get("model_id", type=str, default=None)

    try:
        logger.info("开始重载模型参数。。。")
        for model_type in model_path:
            if model_path[model_type] != None and not model_path[model_type]:
                if os.path.exists(model_path[model_type]):
                    exec(
                        f"model_path[{model_type}] = cat_path(model_path[{model_type}], {model_type}_model_{model_version}_{model_id}.h5)")
                    exec(f"PredictConfig.event_{model_type}_model_path = model_path[{model_type}]")
                    exec(f"get_{model_type}_model()")
                else:
                    logger.error(f"{model_type}模型路径：{model_path[model_type]}错误，请重新传入！")
                    raise ValueError
        logger.info("模型参数重载完成！")
        return jsonify(status="success")

    except:
        trace = traceback.format_exc()
        logger.error(trace)

        return jsonify(status="error", message=trace)


@app.route('/event_extract', methods=['GET', 'POST'])
def extract_infer():
    """
    从前端获取待抽取事件的中文字符串--str
    :return: 返回事件抽取的结果
    """
    # 模型路径字典
    if request.method == "POST":
        query = request.form.get("sentence", type=str, default=None)
    else:
        query = request.args.get("sentence", type=str, default=None)

    # 事件抽取的子队列
    extract_sub_queue = Queue()
    # 向事件抽取工作模块传递待抽取的文本以及子队列
    EXTRACT_QUEUE.put((query, extract_sub_queue))
    success, pred = extract_sub_queue.get()

    if success:
        return jsonify(status="success", data=pred)
    else:
        return jsonify(status="error", message=pred)


@app.route('/coref_with_content', methods=['GET', 'POST'])
def coref_infer():
    """
    从前端获取待消歧的句子--str
    :return: 指代消歧后中文
    """

    if request.method == "POST":
        query = request.form.get("content", type=str, default=None)
    else:
        query = request.args.get("content", type=str, default=None)

    # 指代消解子队列
    coref_sub_queue = Queue()
    # 向指代消解工作模块传入待消解的内容以及指代消解子队列
    COREF_QUEUE.put((query, coref_sub_queue))
    success, pred = coref_sub_queue.get()

    if success:
        return jsonify(status="success", coref=pred)
    else:
        return jsonify(status="error", message=pred)


def extract_work():
    """
    事件抽取主模块，对传入的中文字符串进行事件抽取，将抽取的结果使用子队列传递出去
    :return: None
    """

    def evaluate(sentence):
        """
        传入句子，调用模型对句子中的事件进行抽取，并添加实体信息。
        :param sentence: (str)待抽取事件的句子
        :return: results(list)事件列表
        :raise:TypeError
        """
        if not isinstance(sentence, str) or not sentence:
            logger.error("待抽取事件的句子格式错误，应该使用字符串格式且不能为空！")
            raise TypeError

        # 传入模型和句子进行事件抽取
        events = extract_items(sentence, state_model, trigger_model, object_model, subject_model, loc_model,
                               time_model,
                               negative_model)

        results = []
        for event in events:
            if not event['subject'] and not event['object']:
                continue
            else:
                results.append(event)
        # 对得到的事件列表进行实体信息补充
        results = get_ners_lac(results)

        return results

    while True:

        # 获取数据和子队列
        query, extract_sub_queue = EXTRACT_QUEUE.get()
        try:
            # # 处理文本，抽取事件
            # content = data_process(query)
            content = query
            # sentences = get_sentences(content)
            sentences = [query]
            # 最终的事件结果
            final_result = []
            # 事件id，唯一标识
            event_id = 0
            # 句子id
            sentence_id = 0

            # 遍历句子列表，进行事件抽取
            for sentence in sentences:
                if sentence:
                    events = evaluate(sentence)
                    # 如果句子中没有事件，则给一个各个元素为空的事件列表
                    if not events:
                        events = [{'triggerloc_index': [], 'event_datetime': "", 'event_location': "", 'subject': "",
                                   'verb': "", 'object': "",
                                   'negative_word': "", 'state': "", 'cameo': "",
                                   'namedentity': {'organization': [], 'location': [], 'person': []}}]
                    else:
                        # 获取事件的cameo编号
                        events = get_event_cameo(cameo_model, events)

                    # 给句子中的事件添加id, sentence_id-event_id
                    for event in events:
                        event["event_id"] = f"{sentence_id}-{event_id}"
                        event_id += 1
                    final_result.append(
                        {"sentence": sentence, "sentence_id": str(sentence_id), "events": copy.deepcopy(events)})
                sentence_id += 1

            # 如果没有事件，则给定一个各元素为空的事件列表。
            if not final_result:
                final_result = [{"sentence": content,
                                 "sentence_id": "0",
                                 "events": [
                                     {'event_id': "0-0", 'triggerloc_index': [], 'event_datetime': "",
                                      'event_location': "", 'subject': "", 'verb': "", 'object': "",
                                      'negative_word': "", 'state': "", 'cameo': "",
                                      'namedentity': {'organization': [], 'location': [], 'person': []}}]}]

            # 调用子队列将事件列表返回回去
            extract_sub_queue.put((True, final_result))

        except:
            # 通过子队列发送异常信息
            trace = traceback.format_exc()
            logger.error(trace)
            extract_sub_queue.put((False, trace))

            continue


def coref_work():
    """
    指代消解主模块，将指代结果通过子队列传送出去
    :return: None
    """
    from jdqd.a04.event_extract.algor.predict.xiaoniu_translate import free_transform_any_2_zh, transform_any_2_zh
    from jdqd.a04.event_extract.algor.predict import coref_spacy

    # 加载指代消解模型对象
    nlp = coref_spacy.get_spacy()

    def coref_(query: str):
        """
        对传入的英文字符串进行指代消解，并返回消解后的文本
        指代消解--翻译--中文
        :param query:(str)待消解的英文字符串
        :return:spacy_data(str)指代消解后并翻译成中文的字符串
        :raise:TypeError
        """
        if not isinstance(query, str) or not query:
            logger.error("待消解的内容格式错误，应该是字符串格式，且不能为空！")
            raise TypeError

        # 消除朝鲜和韩国的指代消歧问题，对两个国家进行指代符替换
        # 防止指代消解因朝鲜韩国出现问题，提前将二者替换为NK SK,如果二者同时存在则只替换一个
        if "North Korea" in query:
            query = query.replace("North Korea", "NK")
        elif "South Korea" in query:
            query = query.replace("South Korea", "SK")

        # 指代消解
        spacy_data = coref_spacy.coref_data(nlp, query)

        # 限制字符翻译
        temp = ''
        data = []
        for once in nlp(spacy_data).sents:
            if len(temp) + len(str(once)) < 2000:
                temp = f"{temp} {str(once)}"
            else:
                # 使用免费接口进行翻译，报错则使用收费接口
                trans = free_transform_any_2_zh(temp)
                if type(trans) != str:
                    trans = transform_any_2_zh(temp)
                data.append(trans)
                temp = str(once)

        # 使用免费接口进行翻译，报错则使用收费接口
        trans = free_transform_any_2_zh(temp)
        if type(trans) != str:
            trans = transform_any_2_zh(temp)
        data.append(trans)

        # 将翻译内容拼接为字符串
        spacy_data = "".join(data)

        if "SK" in spacy_data:
            spacy_data = spacy_data.replace("SKn", "韩国").replace("SK", "韩国")
        elif "NK" in spacy_data:
            spacy_data = spacy_data.replace("NKn", "朝鲜").replace("NK", "朝鲜")

        return spacy_data

    while True:

        # 获取数据和子队列
        content, coref_sub_queue = COREF_QUEUE.get()
        try:
            # 进行翻译并进行指代消解
            spacy_data = coref_(content)
            coref_sub_queue.put((True, spacy_data))
        except:
            # 通过子队列发送异常信息
            trace = traceback.format_exc()
            logger.error(trace)
            coref_sub_queue.put((False, trace))

            continue


if __name__ == '__main__':
    # 事件抽取
    t1 = threading.Thread(target=extract_work)
    t1.daemon = True
    t1.start()
    # 指代消解
    t2 = threading.Thread(target=coref_work)
    t2.daemon = True
    t2.start()
    app.run(host='0.0.0.0', port=38082, threaded=True)
