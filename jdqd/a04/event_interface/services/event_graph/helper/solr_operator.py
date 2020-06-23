#!/usr/bin/env python
# -*- coding:utf-8 -*-

from jdqd.a04.event_interface.config.project import Config

config = Config()
fulltext_topN = config.fulltext_topN


def add_document(solr_db, event_id, short_sentence, event_datetime, cameo_code):
    """
    在solr中创建文档，solr使用动态域：graph-*。

    :param solr_db: object.solr连接对象
    :param event_id: string.事件id
    :param short_sentence: string.事件短语
    :param event_datetime: string.事件发生日期
    :param cameo_code: string.CAMEO编号
    :return 文档是否创建成功。
    """
    doc = dict({
        'graph-id': event_id, 'graph-sorten_sentence': short_sentence,
        'graph-event_date': event_datetime, 'graph-cameo_id': cameo_code
    })
    solr_db.add(doc, commit=True)

    return True


def search(solr_db, search, cameo_code, start_date, end_date):
    """
    在solr中根据search进行全文检索，cameo_code、start_date、end_date进行过滤查询。

    :param solr_db: object.solr连接对象
    :param search: string.检索文本
    :param cameo_code: string.CAMEO编号
    :param start_date: string.事件发生开始日期
    :param end_date: string.事件发生结束日期
    :return array，事件id数组。
    """
    query_str = 'graph-sorten_sentence:{search}'.format(search=search)
    flter_str = ''
    if (start_date is not None or end_date is not None) and (start_date != 'None' or end_date != 'None'):
        if (start_date is not None or start_date != '') and (end_date is None or end_date == ''):
            end_date = '*'
        if (start_date is None or start_date == '') and (start_date is not None or end_date != ''):
            start_date = '*'
        if start_date is not None or start_date != '' or end_date is not None or end_date != '':
            flter_str = "graph-event_date:[{start_date} TO {end_date}]".format(
                start_date=start_date, end_date=end_date)

    if cameo_code is not None and cameo_code != '':
        cameo_code = str(cameo_code).split(",")
        if len(cameo_code) > 0:
            if flter_str != '':
                flter_str = flter_str + " AND "
            flter_str = flter_str + "graph-cameo_id:({cameo_ids})".format(cameo_ids=",".join(cameo_code))

    response = solr_db.select(query_str, score=True, sort_order="DESC", start=0, rows=fulltext_topN, fq=flter_str)

    event_id_list = []
    for row in response.results:
        event_id_list.append(row['graph-id'][0])

    return event_id_list
