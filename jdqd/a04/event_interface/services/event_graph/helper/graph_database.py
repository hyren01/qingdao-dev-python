#!/usr/bin/env python
# -*- coding:utf-8 -*-

from jdqd.a04.event_interface.config.project import Config
from jdqd.a04.event_interface.services.event_graph.helper.rdms_database import get_event_rel

config = Config()
global_event_tag = config.global_event_tag


def __node2json(node):
    """
    将图数据库中查询出来的事件节点对象转换为json对象。

    :param node: object.事件节点
    :return json数据。
    """
    json_result = {}
    for key in node.keys():
        json_result[key] = node[key]
    return json_result


def __create_events(graph_db, events, event_relations):
    """
    在图数据库中创建事件节点和关系节点，若事件节点或关系节点传入空数组，则不会根据空数组创建节点。

    :param graph_db: object.图数据库连接对象
    :param events: array.事件节点数据，如：[{"event_name":"", "event_tag":"", "event_attribute":{event_id:'',
                                            event_sentence:'', event_date:''}}]
    :param event_relations: array.关系节点数据，如：[{"source_event_tag":"", "target_event_tag":"",
                                            "source_event_id":"", "target_event_id":"", "realtion":'',
                                            "relation_attribute":{name:''}}]
    :return 事件节点和关系节点是否创建成功。
    """
    tx = graph_db.begin_transaction()
    try:
        for event in events:
            if 'event_tag' in event:
                event_tag = event["event_tag"]
            else:
                event_tag = global_event_tag
            tx.run(
                "CREATE (:{event_tag} {event_attribute})".format(event_tag=event_tag,
                                                                 event_attribute=str(event["event_attribute"]))
            )
        for relation in event_relations:
            if 'source_event_tag' in relation:
                source_event_tag = relation["source_event_tag"]
            else:
                source_event_tag = global_event_tag
            if 'target_event_tag' in relation:
                target_event_tag = relation["target_event_tag"]
            else:
                target_event_tag = global_event_tag
            tx.run(
                ("MATCH (event1:{source_event_tag}),(event2:{target_event_tag}) WHERE "
                 "event1.event_id = '{source_event_id}' AND event2.event_id = '{target_event_id}' "
                 "CREATE (event1)-[:{relation} {relation_attribute}]->(event2)").format(
                    source_event_tag=source_event_tag, target_event_tag=target_event_tag,
                    source_event_id=relation["source_event_id"], target_event_id=relation["target_event_id"],
                    relation=relation["relation"], relation_attribute=relation["relation_attribute"]
                )
            )
        tx.commit()
        return True
    except RuntimeError:
        tx.rollback()
        return False


#
#
# def __create_events(graph_db, events, event_relations):
#     """
#     在图数据库中创建事件节点和关系节点，若事件节点或关系节点传入空数组，则不会根据空数组创建节点。
#
#     :param graph_db: object.图数据库连接对象
#     :param events: array.事件节点数据，如：[{"event_name":"", "event_tag":"", "event_attribute":{event_id:'',
#                                             event_sentence:'', event_date:''}}]
#     :param event_relations: array.关系节点数据，如：[{"source_event_tag":"", "target_event_tag":"",
#                                             "source_event_id":"", "target_event_id":"", "realtion":'',
#                                             "relation_attribute":{name:''}}]
#     :return 事件节点和关系节点是否创建成功。
#     """
#     tx = graph_db.begin_transaction()
#     try:
#         for event in events:
#             if 'event_tag' in event:
#                 event_tag = event["event_tag"]
#             else:
#                 event_tag = global_event_tag
#             tx.run(
#                 "CREATE (:{event_tag} {event_attribute})".format(event_tag=event_tag,
#                                                                  event_attribute=str(event["event_attribute"]))
#             )
#         for relation in event_relations:
#             if 'source_event_tag' in relation:
#                 source_event_tag = relation["source_event_tag"]
#             else:
#                 source_event_tag = global_event_tag
#             if 'target_event_tag' in relation:
#                 target_event_tag = relation["target_event_tag"]
#             else:
#                 target_event_tag = global_event_tag
#             tx.run(
#                 ("MATCH (event1:{source_event_tag}),(event2:{target_event_tag}) WHERE "
#                  "event1.event_id = '{source_event_id}' AND event2.event_id = '{target_event_id}' "
#                  "CREATE (event1)-[:{relation} {relation_attribute}]->(event2)").format(
#                     source_event_tag=source_event_tag, target_event_tag=target_event_tag,
#                     source_event_id=relation["source_event_id"], target_event_id=relation["target_event_id"],
#                     relation=relation["relation"], relation_attribute=relation["relation_attribute"]
#                 )
#             )
#         tx.commit()
#         return True
#     except RuntimeError:
#         tx.rollback()
#         return False


def create_event(graph_db, event_id, short_sentence, event_datetime, event_tag):
    """
    在图数据库中创建事件节点。

    :param graph_db: object.图数据库连接对象
    :param event_id: string.事件id
    :param short_sentence: string.事件短句
    :param event_datetime: string.事件发生日期
    :param event_tag: 事件节点标签（默认Event）
    :return 事件节点是否创建成功。
    """
    if event_tag is None:
        event_tag = global_event_tag
    short_sentence = str(short_sentence).replace("'", "\"")
    event_attribute = "event_id: '{event_id}', event_sentence: '{short_sentence}', event_date: " \
                      "'{event_datetime}'".format(event_id=event_id, short_sentence=short_sentence,
                                                  event_datetime=event_datetime)
    event_attribute = "{" + event_attribute + "}"
    event = [{"event_name": "", "event_tag": event_tag, "event_attribute": event_attribute}]
    success = __create_events(graph_db, event, [])

    return success


def create_relation(graph_db, cause_event_id, effect_event_id, relation_type, event_tag):
    """
    在图数据库中创建关系节点。

    :param graph_db: object.图数据库连接对象
    :param cause_event_id: string.源节点事件id
    :param effect_event_id: string.目前节点事件id
    :param relation_type: string.关系类型
    :param event_tag: 事件节点标签（默认Event）
    :return 关系节点是否创建成功。
    """
    if event_tag is None:
        event_tag = global_event_tag
    relation_attribute = "name: '{name}'".format(name=relation_type)
    relation_attribute = "{" + relation_attribute + "}"
    relation = [{"source_event_tag": event_tag, "target_event_tag": event_tag,
                 "source_event_id": cause_event_id, "target_event_id": effect_event_id,
                 "relation": relation_type, "relation_attribute": relation_attribute}]
    success = __create_events(graph_db, [], relation)
    return success
    # if success != "success":
    #     print("创建关系节点失败")


def get_event_by_ids(graph_db, event_ids, event_tag, start_date, end_date):
    """
    在图数据库中根据事件id数组查询出对应的事件节点。

    :param graph_db: object.图数据库连接对象
    :param event_ids: array.事件id数组，如：['','']
    :param event_tag: string.事件标签（默认Event）
    :param start_date: string.事件发生开始日期
    :param end_date: string.事件发生结束日期
    :return 事件节点数组。
    """
    if event_tag is None:
        event_tag = global_event_tag
    cql_str = "MATCH (event:{event_tag}) WHERE event.event_id IN {ids}".format(event_tag=event_tag, ids=event_ids)
    if start_date is not None and start_date != '' and start_date != 'None':
        cql_str = cql_str + " AND event.event_date > {start_date}".format(start_date=start_date)
    if end_date is not None and end_date != '' and end_date != 'None':
        if start_date is not None and start_date != '' and start_date != 'None':
            cql_str = cql_str + " OR "
        else:
            cql_str = cql_str + " AND "
        cql_str = cql_str + "event.event_date < {end_date}".format(end_date=start_date)
    cql_str = cql_str + " RETURN event"
    results = graph_db.run(cql_str)
    event_result = []
    for record in results:
        event_result.append(__node2json(record["event"]))

    return event_result


def get_event_rel_by_id(graph_db, event_id, event_tag):
    """
    在图数据库中根据事件id查询出对应的跟该事件相关联的所有事件节点及其对应关系。

    :param graph_db: object.图数据库连接对象
    :param event_id: string.事件id
    :param event_tag: string.事件标签（默认Event）

    :return 事件节点及关系数组，如：[{"source_event_id":"", "source_event_sentence":"", "source_event_date":"",
                                      "relation_name":"", "target_event_id":"", "target_event_sentence":"",
                                      "target_event_date":""}]。
    """
    if event_tag is None:
        event_tag = global_event_tag
    cql_str = "MATCH (event1:{event_tag})-[relation]->(event2:{event_tag}) WHERE event1.event_id = '{event_id}' OR " \
              "event2.event_id = '{event_id}' RETURN event1,relation,event2".format(event_tag=event_tag,
                                                                                    event_id=event_id)
    print(cql_str)
    results = graph_db.run(cql_str)
    # 事件关系数据
    result_link = []
    # 事件数据
    result_data = []
    for record in results:
        source_event = __node2json(record["event1"])
        event_relation = __node2json(record["relation"])
        target_event = __node2json(record["event2"])
        result_link.append({'source_event_id': source_event['event_id'],
                            'source_event_sentence': source_event['event_sentence'],
                            'source_event_date': source_event['event_date'],
                            'relation_name': event_relation['name'],
                            'target_event_id': target_event['event_id'],
                            'target_event_sentence': target_event['event_sentence'],
                            'target_event_date': target_event['event_date']})
        # 第一次循环直接添加
        if not result_data:
            get_source_event_rel(result_data, source_event)
            get_target_event_rel(result_data, target_event)
        else:
            # 如果左事件与页面传入的相同，则添加右事件，否则添加左事件
            if source_event['event_id'] == event_id:
                get_source_event_rel(result_data, target_event)
            else:
                get_target_event_rel(result_data, source_event)
    return result_link, result_data


def get_target_event_rel(result_data, target_event):
    # 查询事件属性
    target_event_rel = get_event_rel(target_event['event_id'])
    result_data.append({'id': target_event['event_id'], 'name': target_event['event_sentence'],
                        'event_date': target_event_rel[0]["event_date"],
                        'event_local': target_event_rel[0]["event_local"],
                        'event_state': target_event_rel[0]["event_state"],
                        'nentity_place': target_event_rel[0]["nentity_place"],
                        'nentity_org': target_event_rel[0]["nentity_org"],
                        'nentity_person': target_event_rel[0]["nentity_person"],
                        'nentity_misc': target_event_rel[0]["nentity_misc"]})


def get_source_event_rel(result_data, source_event):
    # 查询事件属性
    source_event_rel = get_event_rel(source_event['event_id'])
    result_data.append({'id': source_event['event_id'], 'name': source_event['event_sentence'],
                        'event_date': source_event_rel[0]["event_date"],
                        'event_local': source_event_rel[0]["event_local"],
                        'event_state': source_event_rel[0]["event_state"],
                        'nentity_place': source_event_rel[0]["nentity_place"],
                        'nentity_org': source_event_rel[0]["nentity_org"],
                        'nentity_person': source_event_rel[0]["nentity_person"],
                        'nentity_misc': source_event_rel[0]["nentity_misc"]})


def get_event_to_event(graph_db, event_tag, event_source_sentence, event_target_sentence):
    if event_tag is None:
        event_tag = global_event_tag
    # 1、图数据库中查询事件关系
    cql_str = f"MATCH (event1:{event_tag})-[relation]->(event2:{event_tag}) " \
              f"WHERE event1.event_sentence = '{event_source_sentence}' OR " \
              f"event2.event_sentence = '{event_target_sentence}' RETURN event1,relation,event2"
    results = graph_db.run(cql_str)
    # 事件与事件的关系
    result_link = []
    # 事件数据
    result_data = []
    # 2、组装数据
    for record in results:
        source_event = __node2json(record["event1"])
        event_relation = __node2json(record["relation"])
        target_event = __node2json(record["event2"])
        result_link.append({'source_event_id': source_event['event_id'],
                            'source_event_sentence': source_event['event_sentence'],
                            'source_event_date': source_event['event_date'],
                            'relation_name': event_relation['name'],
                            'target_event_id': target_event['event_id'],
                            'target_event_sentence': target_event['event_sentence'],
                            'target_event_date': target_event['event_date']})
        # 如果查询出的这2个事件和输入参数一致，则直接添加
        if source_event['event_sentence'] == event_source_sentence and \
                target_event['event_sentence'] == event_target_sentence:
            get_source_event_rel(result_data, source_event)
            get_target_event_rel(result_data, target_event)
        # 否则判断查询出的左事件是否和输入的左事件一致，如果一致添加右事件，否则添加左事件
        else:
            if source_event['event_sentence'] == event_source_sentence:
                get_source_event_rel(result_data, target_event)
            else:
                get_target_event_rel(result_data, source_event)
    return result_link, result_data
