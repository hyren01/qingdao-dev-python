#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
提供向量化事件归并的事件匹配、事件向量化保存、事件删除接口
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import threading
import traceback
from feedwork.utils import logger
from jdqd.common.event_emm import data_utils
from jdqd.common.event_emm.model_utils import TOKENIZER
from jdqd.a04.event_search.algor.predict import load_model, load_vec, save_vec, delete_vec
import jdqd.a04.event_search.config.PredictConfig as pre_config

app = Flask(__name__)
CORS(app)

# bert模型
BERT_MODEL = load_model.load_bert_model()
# 加载匹配模型
MATCH_MODEL = load_model.load_match_model()

# 匹配用的主队列
MATCH_QUEUE = Queue(maxsize=5)
# 事件向量化保存用的主队列
VEC_QUEUE = Queue(maxsize=5)
# 事件删除主队列
VEC_DELETE_QUEUE = Queue(maxsize=5)
# 向量读取主队列
READ_QUENE = Queue(maxsize=5)


def judge(name, data):
    """
    传入数据，判断是否为空
    :param name: 数据名称
    :param data: 数据
    :return: None
    """
    if not data or data is None:
        logger.error(f"{name} is None!")
        raise ValueError


# 持续循环读取向量文件
def vec_reader():
    """
    遍历读取向量文件，持续循环
    :return: None
    """
    while True:
        logger.info(f"READ_QUENE.get是否为空：{READ_QUENE.empty()}", f"是否已经满了{READ_QUENE.full()}")
        message, read_sub_queue = READ_QUENE.get()
        logger.info("READ_QUENE.get完成！")
        logger.info(f"READ_QUENE.get是否为空：{READ_QUENE.empty()}", f"是否已经满了{READ_QUENE.full()}")

        try:
            # 如果需要读取向量则开始读取
            if message:
                # 加载字典文件
                cameo2id = data_utils.read_json(pre_config.cameo2id_path)
                # 获取所有的cameo号
                cameos = list(cameo2id.keys())
                # 一次只读取一个cameo对应的所有向量
                for cameo in cameos:
                    data = load_vec.load_vec_data(cameo)

                    # 向量内容放入读取子队列中
                    logger.info(f"read_sub_queue.put是否为空：{read_sub_queue.empty()}, 是否已经满了{read_sub_queue.full()}")
                    if cameo != cameos[-1]:
                        read_sub_queue.put((True, data))
                    else:
                        read_sub_queue.put((False, data))
                    logger.info("read_sub_queue.put完成！")
            else:
                continue

        except:
            trace = traceback.format_exc()
            logger.error(trace)

            continue


# 事件相似性匹配
@app.route("/event_match", methods=["GET", "POST"])
def event_match():
    """
    事件匹配，从前端获取事件短句、cameo编号、相似度阈值
    :return: 匹配结果给前端
    :raise:ValueError如果短句为空
    """
    if request.method == "GET":
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.args.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.args.get("cameo", type=str, default=None)
        # 相似度阈值
        threshold = request.args.get("threshold", type=float, default=0.5)
    else:
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.form.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.form.get("cameo", type=str, default=None)
        # 相似度阈值
        threshold = request.form.get("threshold", type=float, default=0.5)

    # 默认阈值为0.5
    if not threshold or threshold is None:
        threshold = 0.5


    # 匹配模块子队列
    logger.info("开始创建match_sub_queue")
    match_sub_queue = Queue()
    # 将短句、cameo号、以及阈值传递给匹配模块
    logger.info(f"MATCH_QUEUE.put是否为空：{MATCH_QUEUE.empty()}，是否已经满了{MATCH_QUEUE.full()}")
    MATCH_QUEUE.put((short_sentence, cameo, threshold, match_sub_queue))
    logger.info("MATCH_QUEUE.put完成！")

    # 通过匹配模块获取匹配结果
    logger.info(f"match_sub_queue.get是否为空：{match_sub_queue.empty()}，是否已经满了{match_sub_queue.full()}")
    message, result = match_sub_queue.get()
    logger.info("match_sub_queue.get完成！")
    logger.info(f"match_sub_queue.get是否为空：{match_sub_queue.empty()}，是否已经满了{match_sub_queue.full()}")


    if message:
        return jsonify(status="success", result=result)
    else:
        return jsonify(status="failed", result=result)


def match():
    """
    在相应的cameo中查找最相似的事件
    :return: None
    """
    while True:
        logger.info(f"MATCH_QUEUE.get是否为空:{MATCH_QUEUE.empty()},是否已满{MATCH_QUEUE.full()}")
        short_sentence, cameo, threshold, match_sub_queue = MATCH_QUEUE.get()
        logger.info("MATCH_QUEUE.get完成！")
        logger.info(f"MATCH_QUEUE.get是否为空:{MATCH_QUEUE.empty()},是否已满{MATCH_QUEUE.full()}")

        try:
            # 判断短句是否为空
            judge("short_sentence", short_sentence)

            # 短句向量化
            logger.info("load_model.generate_vec。。。")
            main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
            logger.info("load_model.generate_vec完成！")
            # 最终的结果
            results = []
            # 判断是否全部遍历
            if not cameo or cameo is None or cameo == "":
                logger.info("scaning all files...")
                # 文件读取子队列
                read_sub_queue = Queue()
                # 文件读取主队列
                logger.info(f"READ_QUENE.put是否为空:{READ_QUENE.empty()},是否已满{READ_QUENE.full()}")
                READ_QUENE.put((True, read_sub_queue))
                logger.info("READ_QUENE.put完成！")

                # 从向量
                while True:
                    logger.info(f"read_sub_queue.get是否为空:{read_sub_queue.empty()},是否已满{read_sub_queue.full()}")
                    status, data = read_sub_queue.get()
                    logger.info("read_sub_queue.get完成！")
                    logger.info(f"read_sub_queue.get是否为空:{read_sub_queue.empty()},是否已满{read_sub_queue.full()}")

                    if status:
                        for once in data:
                            score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                            if score >= threshold:
                                results.append({"event_id": once, "score": float(score)})
                    else:
                        for once in data:
                            score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                            if score >= threshold:
                                results.append({"event_id": once, "score": float(score)})
                        break
                results.sort(key=lambda x: x["score"], reverse=True)

            else:
                # 加载数据
                logger.info("scaning identied cameo file!")
                data = load_vec.load_vec_data(cameo)
                logger.info("identied cameo file is OK!")
                # 如果data不为空则执行此处操作，否则就任务数据文件为空，这是第一条数据
                if data:
                    for once in data:
                        score = load_model.vec_match(main_vec, data[once], MATCH_MODEL)
                        if score >= threshold:
                            results.append({"event_id": once, "score": float(score)})
                    # 对输出的结果按照降序排序
                    results.sort(key=lambda x: x["score"], reverse=True)

            # 将匹配结果返回给接口
            logger.info(f"match_sub_queue.put是否为空:{match_sub_queue.empty()},是否已满{match_sub_queue.full()}")
            match_sub_queue.put((True, results))
            logger.info("match_sub_queue.put完成！")

        except:
            trace = traceback.format_exc()
            logger.error(trace)

            logger.info(f"EXCEPT:match_sub_queue.put是否为空:{match_sub_queue.empty()},是否已满{match_sub_queue.full()}")
            match_sub_queue.put((False, trace))
            logger.info("EXCEPT:match_sub_queue.put完成！")

            continue


# 事件向量化保存
@app.route("/vec_save", methods=["GET", "POST"])
def event_vectorization():
    """
    事件向量化接口，从前端接收短句、cameo编号、事件id,将事件向量化保存到文件中。
    :return: 保存状态
    :raise:事件短句、事件id为空--ValueError
    """
    if request.method == "GET":
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.args.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.args.get("cameo", type=str, default=None)
        # 事件id
        event_id = request.args.get("event_id", type=str, default=None)
    else:
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.form.get("short_sentence", type=str, default=None)
        # 事件cameo号
        cameo = request.form.get("cameo", type=str, default=None)
        # 事件id
        event_id = request.form.get("event_id", type=str, default=None)

    # 向量化模块子队列
    vec_sub_queue = Queue()
    # 将短句、cameo号、以及事件id传递给向量化模块
    VEC_QUEUE.put((short_sentence, cameo, event_id, vec_sub_queue))
    # 获取执行状态
    status, message = vec_sub_queue.get()

    if status:
        return jsonify(status="success", message=message)
    else:
        return jsonify(status="failed", message=message)


def vec_save():
    """
    从前端获取事件短句、cameo号、事件id，将事件短句向量化并保存
    :return: None
    """
    while True:
        # 从接口处获取事件短句、cameo编号、事件id、子队列
        short_sentence, cameo, event_id, vec_sub_queue = VEC_QUEUE.get()
        try:
            # 判断短句是否为空
            judge("short_sentence", short_sentence)
            # 判断cameo是否为空
            judge("cameo", cameo)
            # 判断事件id是否为空
            judge("event_id", event_id)
            # 事件短句向量化
            main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
            # 向量保存
            save_vec.save_vec_data(cameo, event_id, main_vec)
            # 返回状态值
            vec_sub_queue.put((True, ""))
        except:
            trace = traceback.format_exc()
            logger.error(trace)
            vec_sub_queue.put((False, trace))
            continue


# 事件向量删除
@app.route("/vec_delete", methods=["GET", "POST"])
def vev_delete():
    """
    事件向量化接口，从前端接收事件id,将事件从事件cameo字典以及npy文件中删除。
    :return: 删除状态
    :raise:事件id为空--ValueError
    """
    if request.method == "GET":
        # 事件id
        event_id = request.args.get("event_id", type=str, default=None)
    else:
        # 事件id
        event_id = request.form.get("event_id", type=str, default=None)

    # 向量化模块子队列
    vec_delete_sub_queue = Queue()
    # 事件id传递给向量删除模块
    VEC_DELETE_QUEUE.put((event_id, vec_delete_sub_queue))
    # 获取执行状态
    status, message = vec_delete_sub_queue.get()

    if status:
        return jsonify(status="success", message=message)
    else:
        return jsonify(status="failed", message=message)


def vec_delete_execute():
    """
    通过队列接收待删除的事件id,遍历cameo_id列表，将id以及对应的事件向量删除。
    :return: None
    """
    while True:
        event_id, vec_delete_sub_queue = VEC_DELETE_QUEUE.get()

        try:
            # 判断短句是否为空
            judge("event_id", event_id)
            logger.info("Begin to delete vector...")
            # 删除事件id以及对应的向量
            state = delete_vec.execute_delete(event_id)

            # 如果state为1则删除成功,0则没有找到对应的event_id
            if state:
                vec_delete_sub_queue.put((True, "success"))
            else:
                vec_delete_sub_queue.put((False, "Event_id not in saved file!"))

        except:
            trace = traceback.format_exc()
            logger.error(trace)
            vec_delete_sub_queue.put((False, trace))
            continue


if __name__ == "__main__":

    # 向量遍历线程
    t0 = threading.Thread(target=vec_reader)
    t0.daemon = True
    t0.start()
    # 向量匹配线程
    t1 = threading.Thread(target=match)
    t1.daemon = True
    t1.start()
    # 向量保存线程
    t2 = threading.Thread(target=vec_save)
    t2.daemon = True
    t2.start()
    # 向量删除线程
    t3 = threading.Thread(target=vec_delete_execute)
    t3.daemon = True
    t3.start()
    app.run(host="0.0.0.0", port="38083", threaded=True)
