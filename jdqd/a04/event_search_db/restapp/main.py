#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
提供向量化事件归并的事件匹配、事件向量化保存、事件删除接口
"""
import time
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from feedwork.utils import logger
from jdqd.common.event_emm.model_utils import TOKENIZER
from jdqd.a04.event_search_db.algor.predict import load_model, load_vec, save_vec, delete_vec, db_connect

app = Flask(__name__)
CORS(app)

# 创建数据库连接
DB = db_connect.get_connection()
# 加载向量数据
logger.info(f"开始加载向量数据。。。")
VEC_DATA = load_vec.load_vec_data(DB)
logger.info(f"向量数据加载完成！")

# 加载cameo字典
logger.info(f"开始加载CAMEO字典。。。")
CAMEO2ID = load_vec.load_vec_data(DB)
logger.info(f"CAMEO字典加载完成！")

# bert模型
BERT_MODEL = load_model.load_bert_model()
# 加载匹配模型
MATCH_MODEL = load_model.load_match_model()


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

    try:
        # 判断短句是否为空
        judge("short_sentence", short_sentence)
        # 短句向量化
        main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
        # 最终的结果
        results = []
        for once in CAMEO2ID[cameo]:
            score = load_model.vec_match(main_vec, VEC_DATA[once], MATCH_MODEL)
            if score >= threshold:
                results.append({"event_id": once, "score": float(score)})
        # 对输出的结果按照降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify(status="success", result=results)
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        return jsonify(status="failed", result=trace)


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
        save_vec.save_vec_data(DB, cameo, event_id, main_vec)
        # 加载到向量字典中
        VEC_DATA[event_id] = main_vec
        event_ids = CAMEO2ID.setdefault(cameo,[])
        event_ids.append(event_id)
        return jsonify(status="success", message="success")
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        return jsonify(status="failed", message=str(trace))


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

    try:
        # 判断短句是否为空
        judge("event_id", event_id)
        # 删除事件id以及对应的向量
        delete_vec.execute_delete(DB, event_id)
        # 删除向量字典中的向量
        del VEC_DATA[event_id]
        for cameo in CAMEO2ID:
            if event_id in CAMEO2ID[cameo]:
                CAMEO2ID[cameo].remove(event_id)

        return jsonify(status="success", message="success")
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        jsonify(status="failed", message=str(trace))


@app.route("/search", methods=["GET", "POST"])
def event_search():
    """
    供前端事件检索使用，返回top_n
    :return: results(list)-->[{"event_id":"", "score":}]
    """
    if request.method == "GET":
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.args.get("short_sentence", type=str, default=None)
        # 相似度阈值
        threshold = request.args.get("threshold", type=float, default=0.5)
        # top_n
        top_n = request.args.get("top_n", type=int, default=50)
    else:
        # 事件短句，由主语、谓语、否定词、宾语组成
        short_sentence = request.form.get("short_sentence", type=str, default=None)
        # 相似度阈值
        threshold = request.form.get("threshold", type=float, default=0.5)
        # top_n
        top_n = request.args.get("top_n", type=int, default=50)

    try:
        # 必须是整型
        if not top_n or top_n == None or type(top_n) != int:
            logger.error(f"top_n 类型错误！")
            raise TypeError
        # 判断短句是否正常
        judge("short_sentence", short_sentence)

        t1 = time.time()
        # 使用bert向量化
        main_vec = load_model.generate_vec(TOKENIZER, BERT_MODEL, short_sentence)
        t2 = time.time()

        logger.info(f"向量化耗时：{t2 - t1}s")

        if VEC_DATA:
            # 使用余弦值方法计算相似度
            all_vecs = list(VEC_DATA.values())
            scores = cosine_similarity(all_vecs, [main_vec]).reshape(-1)
            results = [{"event_id": once, "score": float(score)} for once, score in zip(VEC_DATA, scores) if
                       score >= threshold]
        else:
            results = []

        t3 = time.time()
        logger.info(f"计算相似度耗时：{t3 - t2}s")
        # 对输出的结果按照降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        # 取top_n
        if len(results) > top_n:
            results = results[:top_n]
        t4 = time.time()
        logger.info(f"排序耗时：{t4 - t3}s")

        # 返回匹配结果
        return jsonify(status="success", result=results)
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        return jsonify(status="failed", result=trace)


if __name__ == "__main__":

    app.run(host="0.0.0.0", port="38083", threaded=True)
