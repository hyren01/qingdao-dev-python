#!/usr/bin/env python
# -*- coding:utf-8 -*-
from jdqd.a04.event_interface.config.project import Config
from jdqd.a04.event_interface.services.common.gener_id import gener_id_by_uuid
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
import feedwork.utils.DateHelper as date_util
from feedwork.utils import logger

config = Config()


def insert_event_sentence(article_id, event_sentence):
    """
    插入数据到“事件句子表”。

    :param article_id: string.文章编号
    :param event_sentence: string.事件句子
    :return 文本编号、句子编号。
    """
    db = DatabaseWrapper()
    try:
        event_sentence = str(event_sentence).replace("'", "\"")
        sentence_id = gener_id_by_uuid()
        if article_id is None or article_id == '':
            article_id = gener_id_by_uuid()
        # cursor = connect.cursor()
        db.execute("INSERT INTO ebm_event_sentence(sentence_id, event_sentence, article_id) VALUES "
                   "(%s, %s, %s)", (sentence_id, event_sentence, article_id))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()
    return article_id, sentence_id


def insert_event_info(subject, verb, object, shorten_sentence, cameo_code, triggerloc_index, event_negaword):
    """
    插入数据到“事件信息表”。

    :param subject: string.主语
    :param verb: string.谓语
    :param object: string.宾语
    :param shorten_sentence: string.事件短语
    :param cameo_code: string.cameo编号
    :param triggerloc_index: string.谓语下标
    :param event_negaword: string.否定词
    :return 事件id。
    """
    db = DatabaseWrapper()
    try:
        triggerloc_index = str(triggerloc_index).replace("'", "\"")
        event_id = gener_id_by_uuid()
        # cursor = connect.cursor()
        db.execute("INSERT INTO ebm_event_info(event_id, subject, verb, object, shorten_sentence, cameo_id, "
                   "triggerloc_index, event_negaword) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                   (event_id, subject, verb, object, shorten_sentence, cameo_code, triggerloc_index, event_negaword))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()

    return event_id


def insert_event_copy(copy_event_id, event_id):
    """
    插入数据到“事件信息镜像校验表”。

    :param copy_event_id: string.镜像事件编号
    :param event_id: string.事件编号
    :return 镜像id。
    """
    db = DatabaseWrapper()
    try:
        copy_id = gener_id_by_uuid()
        # cursor = connect.cursor()
        db.execute("INSERT INTO ebm_event_copy(copy_id, copy_event_id, event_id) VALUES (%s, %s, %s)",
                   (copy_id, copy_event_id, event_id))
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()

    return copy_id


def insert_event_attribute(sentiment_analysis, event_date, event_local, event_state, nentity_place,
                           nentity_org, nentity_person, nentity_misc, event_id):
    """
    插入数据到“事件属性表”。

    :param sentiment_analysis: string.情感分析
    :param event_date: string.事件发生日期
    :param event_local: string.事件发生地点
    :param event_state: string.事件状态
    :param nentity_place: string.命名实体-地点
    :param nentity_org: string.命名实体-组织机构
    :param nentity_person: string.命名实体-人
    :param nentity_misc: string.命名实体-杂项
    :param event_id: string.事件编号
    :return 属性id。
    """
    db = DatabaseWrapper()
    try:
        if not nentity_place:
            nentity_place = ''
        if not nentity_org:
            nentity_org = ''
        if not nentity_person:
            nentity_person = ''
        event_local = str(event_local).replace("'", "\"")
        nentity_place = str(nentity_place).replace("'", "\"")
        nentity_org = str(nentity_org).replace("'", "\"")
        nentity_person = str(nentity_person).replace("'", "\"")
        attritute_id = gener_id_by_uuid()
        create_date = date_util.sys_datetime("%Y-%m-%d")
        # cursor = connect.cursor()
        db.execute("INSERT INTO ebm_eventsent_rel(relation_id, sentiment_analysis, event_date, event_local, "
                   "event_state, nentity_place, nentity_org, nentity_person, nentity_misc, "
                   "create_date, event_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                   (attritute_id, sentiment_analysis, event_date, event_local, event_state, nentity_place,
                    nentity_org, nentity_person, nentity_misc, create_date, event_id))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()

    return attritute_id


def insert_sentattribute_rel(shorten_ssentence, attribute_id, sentence_id):
    """
    插入数据到“事件句子属性关系表”。

    :param shorten_ssentence: string.事件原始短语。该数据在事件归并时才有效，用于记录相似事件的事件短语。
    :param attribute_id: string.情感分析
    :param sentence_id: string.事件发生日期
    :return 关系id。
    """
    db = DatabaseWrapper()
    try:
        rel_id = gener_id_by_uuid()
        # cursor = connect.cursor()
        db.execute("INSERT INTO ebm_sentattribute_rel(rel_id, shorten_ssentence, relation_id, sentence_id) VALUES "
                   "(%s, %s, %s, %s)", (rel_id, shorten_ssentence, attribute_id, sentence_id))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()

    return rel_id


def get_cameoinfo_by_id(event_id):
    """
    根据事件id查询该事件的CAMEO信息。

    :param event_id: string.事件id
    :return CAMEO详细信息。
    """
    db = DatabaseWrapper()
    try:
        # cursor = connect.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        result = db.query(f"SELECT t1.*, t2.* FROM ebm_event_info t1 "
                          f"JOIN ebm_cameo_info t2 ON t1.cameo_id = t2.cameo_id "
                          f"WHERE t1.event_id = '{event_id}'", (), QueryResultType.JSON)
        # result = cursor.fetchall()
        # result = json.loads(json.dumps(result))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def get_sentence_attributes_by_id(event_id):
    """
    根据事件id查询该事件的详细信息。

    :param event_id: string.事件id
    :return 事件的详细信息。
    """
    db = DatabaseWrapper()
    try:
        # cursor = connect.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        result = db.query(f"SELECT t1.*, t2.event_sentence, t2.article_id FROM ebm_eventsent_rel t1 "
                          f"JOIN ebm_event_sentence t2 ON t1.sentence_id = t2.sentence_id "
                          f"WHERE t1.event_id = '{event_id}'", (), QueryResultType.JSON)
        # result = cursor.fetchall()
        # result = json.loads(json.dumps(result))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def get_article_info_by_id(event_id):
    """
    根据事件id查询该事件的文章信息。

    :param event_id: string.事件id
    :return 事件的文章信息。
    """
    db = DatabaseWrapper()
    try:
        # cursor = connect.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        result = db.query(f"SELECT t4.translated_title, t4.translated_content, t4.pub_time, t4.create_date FROM "
                          f"(SELECT t2.article_id FROM ebm_eventsent_rel t1 JOIN ebm_event_sentence t2 ON "
                          f"t1.sentence_id = t2.sentence_id WHERE t1.event_id = '{event_id}' "
                          f"GROUP BY t2.article_id) t3 "
                          f"JOIN t_article_msg t4 ON t3.article_id = t4.article_id", (), QueryResultType.JSON)
        # result = cursor.fetchall()
        # result = json.loads(json.dumps(result))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def get_synonym():
    """
    查询“同义词表”获取同义词数据。

    :return 同义词数据。
    """
    db = DatabaseWrapper()
    try:
        # cursor = connect.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        result = db.query("SELECT * FROM ebm_synonym_storage", (), QueryResultType.JSON)
        # result = cursor.fetchall()
        # result = json.loads(json.dumps(result))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def get_event_info(subject, verb, object, event_negaword):
    """
    根据主语、谓语、宾语查询“事件信息表”。

    :param subject: array.主语
    :param verb: array.谓语
    :param object: array.宾语
    :param event_negaword: string.同义词
    :return 事件信息表数据。
    """
    db = DatabaseWrapper()
    try:
        # cursor = connect.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = f"SELECT * FROM ebm_event_info WHERE subject IN %s AND verb IN %s " \
              f"AND object IN %s AND event_negaword = %s"
        logger.info(sql)
        result = db.query(sql, (tuple(subject), tuple(verb), tuple(object), event_negaword),
                          QueryResultType.JSON)
        # cursor.execute(sql)
        # result = cursor.fetchall()
        # result = json.loads(json.dumps(result))
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def get_event_attribute(event_id, event_local, event_date, nentity_org, nentity_person):
    """
    根据事件id及事件发生地点、组织机构、人物查询事件信息，主要用于验证是否有重复事件。

    :param event_id: string.事件id
    :param event_local: string.事件发生地点
    :param event_date: string.事件发生日期
    :param nentity_org: list.命名实体-组织机构
    :param nentity_person: list.命名实体-人物
    :return 事件信息表数据。
    """
    db = DatabaseWrapper()
    try:
        check_sql = f"SELECT * FROM ebm_eventsent_rel WHERE event_id = '{event_id}' " \
                    f"AND event_local = '{event_local}' " \
                    f"AND event_date = '{event_date}'"
        for org in nentity_org:
            check_sql = check_sql + f" AND (nentity_org LIKE '%%,{org}%%' " \
                                    f"OR nentity_org LIKE '%%{org},%%' OR " \
                                    f"nentity_org = '{org}')"
        for person in nentity_person:
            check_sql = check_sql + f" AND (nentity_person LIKE '%%,{person}%%' " \
                                    f"OR nentity_person LIKE '%%{person},%%' " \
                                    f"OR nentity_person = '{person}')"
        result = db.query(check_sql, (), QueryResultType.JSON)
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()

    return result


def insert_zhcontent(article_id, content):
    """
    插入数据到“t_article_msg_zh”表。

    :param article_id: string.文章id
    :param content: string.文本内容
    """
    db = DatabaseWrapper()
    try:
        db.execute("INSERT INTO t_article_msg_zh(article_id, content) VALUES (%s, %s)", (article_id, content))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()


def insert_corecontent(article_id, content):
    """
    插入数据到“t_article_msg_zh”表。

    :param article_id: string.文章id
    :param content: string.文本内容
    """
    db = DatabaseWrapper()
    try:
        db.execute("INSERT INTO t_article_msg_en(article_id, content) VALUES (%s, %s)", (article_id, content))
        db.commit()
    except Exception as e:
        db.rollback()
        raise RuntimeError(e)
    finally:
        db.close()


def get_article_list(event_id):
    db = DatabaseWrapper()
    article = []
    try:
        if event_id != ['']:
            # 根据多个事件id,查询文章列表
            article_ids = db.query(f"select distinct(a.article_id) from ebm_event_sentence a "
                                   f"join ebm_sentattribute_rel b on a.sentence_id=b.sentence_id "
                                   f"join ebm_eventsent_rel c on b.relation_id=c.relation_id "
                                   f"join ebm_event_info d on c.event_id=d.event_id "
                                   f"where d.event_id in (%s)" % ','.join(['%s'] * len(event_id)), event_id,
                                   QueryResultType.PANDAS)
            article_id_arr = []
            for ids in article_ids.article_id:
                article_id_arr.append(ids)
            # 查询文章信息
            article = db.query(
                "select article_id,title as translated_title,title from t_article_msg_zh where article_id in (%s)" % ','.join(
                    ['%s'] * len(article_id_arr)), article_id_arr, QueryResultType.JSON)
            # 查询文章关联的事件，用于前端连线
            for a in article:
                article_event = db.query(f"select distinct(a.event_id) from ebm_event_info a "
                                         f"join ebm_eventsent_rel b on a.event_id=b.event_id "
                                         f"join ebm_sentattribute_rel c on b.relation_id=c.relation_id "
                                         f"join ebm_event_sentence d on c.sentence_id=d.sentence_id "
                                         f"where d.article_id='{a['article_id']}'", (), QueryResultType.JSON)
                a.update({"event_ids": article_event})

    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()
    return article


def get_event_rel(event_id):
    db = DatabaseWrapper()
    try:
        # 查询事件属性
        event_rel = db.query(f"select * from ebm_eventsent_rel where event_id='{event_id}'", (), QueryResultType.DICT)
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()
    return event_rel


def get_article_info(article_id, event_id):
    db = DatabaseWrapper()
    try:
        # 查询指代消解后的文章详情
        article = db.query(f"select content from t_article_msg_zh where article_id='{article_id}'", (),
                           QueryResultType.JSON)
        # 查询事件关联的句子，用于页面高亮
        sentence = db.query(f"select distinct(a.event_sentence) from ebm_event_sentence a "
                            f"join ebm_sentattribute_rel b on a.sentence_id=b.sentence_id "
                            f"join ebm_eventsent_rel c on b.relation_id=c.relation_id "
                            f"join ebm_event_info d on c.event_id=d.event_id "
                            f"where a.article_id='{article_id}' "
                            f"and d.event_id in (%s)" % ','.join(['%s'] * len(event_id)),
                            event_id, QueryResultType.JSON)
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()
    return article, sentence


def get_article():
    db = DatabaseWrapper()
    try:
        article = db.query(f"select article_id,content from t_article_msg "
                           f"where is_clean='0'")

        return article
    except Exception as e:
        raise RuntimeError(e)
    finally:
        db.close()
    return article

# if __name__ == '__main__':
#     # db = Database()
#     # connect = db.get_connection()
#     # result_db = get_article_info_by_id(connect, "2f2b6ffa6f4411eab165b052160f9f84")
#     # connect.close()
