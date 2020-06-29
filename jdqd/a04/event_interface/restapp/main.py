#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json

import solr
from flask import Flask, g, Response, request
from neo4j import GraphDatabase, basic_auth
from jdqd.a04.event_interface.services.event_graph.helper import event_helper
from jdqd.a04.event_interface.config.project import Config
from jdqd.a04.event_interface.services.common.data_util import clear_web_data

import jdqd.a04.event_interface.services.event_graph.helper.graph_database as graph_service
import jdqd.a04.event_interface.services.event_graph.helper.rdms_database as rdms_service

# import services.solr_operator as solr_service

app = Flask(__name__)

config = Config()
gdb_driver = GraphDatabase.driver(config.neo4j_uri, auth=basic_auth(config.neo4j_username, config.neo4j_password))


def get_gdb():
    """
    开启图数据库链接
    """
    if not hasattr(g, 'neo4j_db'):
        g.neo4j_db = gdb_driver.session()
    return g.neo4j_db


# def get_db():
#     if not hasattr(g, 'postgres_db'):
#         g.postgres_db = psycopg2.connect(host=config.db_host, port=config.db_port, database=config.db_name,
#                                          user=config.db_user, password=config.db_passwd)
#     return g.postgres_db


# def get_solr():
#     if not hasattr(g, 'solr'):
#         g.solr = solr.Solr(config.solr_uri)
#     return g.solr


# @app.route("/create_events", methods=['POST'])
# def create_events():
#     """
#     创建事件的接口
#     :return:
#     """
#     try:
#         events = request.form.get('events')
#         event_relations = request.form.get('event_relations')
#         if events is None:
#             return {"status": "error", "msg": "events参数不能为空"}
#     except KeyError:
#         return {"status": "error"}
#     else:
#         events = json.loads(events)
#         event_relations = json.loads(event_relations)
#         graph_db = get_gdb()
#         success = graph_service.create_event(graph_db, events, event_relations)
#
#         if success:
#             return {"status": "success"}
#         else:
#             return {"status": "error", "msg": "在图数据库中创建事件失败"}


@app.route("/get_events_by_ids", methods=['POST'])
def get_event_by_ids():
    """
    根据事件id数组，查询出事件节点数组。
    传入参数，如：['2963f386725811ea9312b052160f9f84', '112']
    返回数据，如：{"status": "success", "events": [{"event_sentence": "", "event_id": "", "event_date": ""}]}
    """
    try:
        event_ids = request.form.get('event_ids')
        event_tag = request.form.get('event_tag')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        if event_ids is None:
            return {"status": "error", "msg": "event_ids参数为None"}
    except KeyError:
        return {"status": "error"}
    else:
        graph_db = get_gdb()
        event_result = graph_service.get_event_by_ids(graph_db, event_ids, event_tag, start_date, end_date)

        return Response(json.dumps({"status": "success", "events": event_result}), mimetype="application/json")


# @app.route("/solr_search", methods=['POST'])
# def solr_search():
#     """
#     文本相似度匹配接口。
#     传入参数，如：{"search":""}
#     返回数据，如：{"status": "success", "event_ids": ["28c8994a725811ea88d7b052160f9f84"]}
#     """
#     try:
#         search = request.form.get('search')
#         cameo_code = request.form.get('cameo_code')
#         start_date = request.form.get('start_date')
#         end_date = request.form.get('end_date')
#         # 1、程序对用户输入做合法性验证，验证搜索句子是否为空、事件发生日期及其范围是否合法；
#         if search is None:
#             return {"status": "error", "msg": "search参数为None"}
#         if start_date is not None and end_date is not None and end_date > start_date:
#             return {"status": "error", "msg": "结束日期大于开始日期"}
#     except KeyError:
#         return {"status": "error"}
#     else:
#         solr_db = get_solr()
#         event_id_list = solr_service.search(solr_db, search, cameo_code, start_date, end_date)
#
#         return Response(json.dumps({"status": "success", "event_ids": event_id_list}), mimetype="application/json")


@app.route("/get_events_by_search", methods=['POST'])
def get_events_by_search():
    """
    事件检索接口。
    传入参数，如：{"search":"", "start_date":"", "end_date":""}
    返回数据，如：{"status":"success", "events":[{"event_sentence":"", "event_id":"", "event_date":""}]}
    """
    try:
        search = request.form.get('search')
        event_tag = request.form.get('event_tag')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        # 1、程序对用户输入做合法性验证，验证搜索句子是否为空、事件发生日期及其范围是否合法；
        if search is None:
            return {"status": "error", "msg": "search参数为None"}
        if start_date is not None and end_date is not None and end_date > start_date:
            return {"status": "error", "msg": "结束日期大于开始日期"}
    except KeyError:
        return {"status": "error"}
    else:
        status, events = event_helper.get_events_by_search_helper(search, event_tag, start_date, end_date)

        if status != 'success':
            return {"status": "error", "msg": "在根据事件id数组获取事件时发生异常"}

        return Response(json.dumps({"status": "success", "events": events}), mimetype="application/json")


@app.route("/get_eventdetail_by_id", methods=['POST'])
def get_eventdetail_by_id():
    """
    事件详情查询接口。
    传入参数，如：{"event_id":""}
    返回数据，如：{"status":"success", "cameo_info":{"event_id":"", "subject":"", "verb":"", "object":"",
                   "shorten_sentence":"", "cameo_id":"", "event_name":"", "event_descrition":"", "use_instruction":"",
                   "example1":"", "example2":"", "example3":""}, "article_infos":[{"carticle_title":"",
                   "carticle_content":"", "article_date":"", "create_date":""}],
                   "sentence_attributes":[{"relation_id":"", "sentiment_analysis":"", "event_date":"", "event_local":"",
                   "event_negaword":"", "event_state":"", "nentity_place":"", "nentity_org":"", "nentity_person":"",
                   "nentity_misc":"", "create_date":"", "event_id":"", "sentence_id":"", "event_sentence":"",
                   "article_id":""}] }
    """
    try:
        event_id = request.form.get('event_id')
        # 1、程序对用户输入做合法性验证，验证事件id是否为空；
        if event_id is None:
            return {"status": "error", "msg": "event_id参数为None"}
    except KeyError:
        return {"status": "error"}
    else:
        # 2、程序“调用RDMS数据库查询接口”，在RDMS数据库中根据事件id查询出文章详细信息数组以及事件详情数组；
        # connect = get_db()

        event_and_cameo_info = rdms_service.get_cameoinfo_by_id(event_id)
        if len(event_and_cameo_info) < 1:
            event_and_cameo_info = ""
        else:
            event_and_cameo_info = event_and_cameo_info[0]
        sentence_attributes = rdms_service.get_sentence_attributes_by_id(event_id)
        article_infos = rdms_service.get_article_info_by_id(event_id)

        return Response(json.dumps({"status": "success", "cameo_info": event_and_cameo_info,
                                    "article_infos": article_infos, "sentence_attributes": sentence_attributes}),
                        mimetype="application/json")


@app.route("/get_event_rel_by_id", methods=['POST'])
def get_event_rel_by_id():
    """
    事件关系查询接口。
    传入参数，如：{"event_id":"", "event_tag":""}
    返回数据，如：{"status":"success", "source_event":{"event_sentence":"", "event_id":"", "event_date":""},
                   "relations":[{"name":""}], "target_events":[{"event_sentence":"", "event_id":"", "event_date":""}]}
    """
    try:
        event_id = request.form.get('event_id')
        event_tag = request.form.get('event_tag')
        if event_id is None:
            return {"status": "error", "msg": "event_id参数为None"}
    except KeyError:
        return {"status": "error"}
    else:
        graph_db = get_gdb()
        result_link, result_data = graph_service.get_event_rel_by_id(graph_db, event_id, event_tag)

        return Response(json.dumps({"status": "success", "result_link": result_link, "result_data": result_data}),
                        mimetype="application/json")


@app.route("/event_parsed_extract", methods=['POST'])
def event_parsed_extract():
    """
    事件解析及存储接口，该接口对传入的一篇文章或一个句子进行事件分析和抽取，中间数据会入库。
    传入参数，如：{"content":""}
    返回数据，如：{"status":"success", "content_id":""}
    """
    # 1、程序验证输入的原始语种字符串是否为空，文章id是否为空；
    try:
        content = request.form.get('content')
        content_id = request.form.get('content_id')
        event_tag = request.form.get('event_tag')
        update_synonyms = request.form.get('update_synonyms')
        if content is None:
            return {"status": "error", "msg": "content参数为None"}
        # if content_id is None:
        #     return {"status": "error", "msg": "content_id参数为None"}
        if update_synonyms is None:
            update_synonyms = False
    except KeyError:
        return {"status": "error"}
    else:
        # rdms_db = get_db()
        graph_db = get_gdb()
        # fulltext_db = get_solr()
        success, event_id_list, event_list = event_helper.event_parsed_extract_helper(graph_db, content,
                                                                                      content_id, event_tag,
                                                                                      update_synonyms)
        if success:
            success = "success"
        else:
            success = "error"

        # 14、接口返回内容id（文章或者句子的id）。
        return {"status": success, "event_id_list": event_id_list, "event_list": event_list}


@app.route("/clear_web_data", methods=['POST'])
def clear_data_with_web():
    """
    清理web数据接口。
    传入参数，如：{"content":""}
    返回数据，如：{"status":"success", "content":""}
    """
    # 1、程序验证输入的原始语种字符串是否为空，文章id是否为空；
    try:
        content = request.form.get('content')
        if content is None:
            return {"status": "error", "msg": "content参数为None"}
    except KeyError:
        return {"status": "error"}
    else:
        content = clear_web_data(content)

        return {"status": "success", "content": content}


@app.route("/get_event_rel_and_article", methods=['POST'])
def get_article_list():
    """
    获取首页文章列表以及文章和事件的关系
    传入参数：多个事件id
    :return:
    """
    try:
        event_id = request.values.getlist('event_id')
    except KeyError:
        return {"status": "error"}
    else:
        result_article = event_helper.get_article_list(event_id)

        return Response(json.dumps({"status": "success", "result": result_article}), mimetype="application/json")


@app.route("/get_article_info", methods=['POST'])
def get_article_info():
    """
    查询文章详情
    传入参数：文章id，多个事件id
    :return:
    """
    try:
        article_id = request.form.get('article_id')
        event_id = request.args.getlist('event_id')
    except KeyError:
        return {"status": "error"}
    else:
        result_article, result_sentence = event_helper.get_article_info(article_id, event_id)

        return Response(
            json.dumps({"status": "success", "result_article": result_article[0], "result_sentence": result_sentence}),
            mimetype="application/json")


@app.route("/get_event_to_event", methods=['POST'])
def get_event_to_event():
    """
    根据左事件和右事件，在图数据库中查询关系
    传入参数：左事件句子，右事件句子
    :return: result_data 关系信息
    """
    try:
        event_source_sentence = request.form.get('event_source_sentence')
        event_target_sentence = request.form.get('event_target_sentence')
        event_tag = request.form.get('event_tag')
    except KeyError:
        return {"status": "error"}
    else:
        graph_db = get_gdb()
        result_link, result_data = event_helper.get_event_to_event(graph_db, event_tag, event_source_sentence,
                                                                   event_target_sentence)

        # 14、接口返回内容id（文章或者句子的id）。
        return Response(json.dumps({"status": "success", "result_data": result_data, "result_link": result_link}),
                        mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.http_port)
