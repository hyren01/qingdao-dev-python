import requests
import json
import re
from jdqd.a04.relation.services.common.rdms_db_util import insert_event_relations, get_event_info
from jdqd.a04.relation.services.common.graph_db_util import insert_graph_db
from itertools import product
from feedwork.utils import logger
from jdqd.a04.relation.config.services import Config
from jdqd.a04.relation.services.common.get_sentences import get_sentences
from jdqd.a04.relation.relation_pt.services.split_server import split
from jdqd.a04.relation.relation_extract.algor.predict.relation_class_pre import class_pre
from jdqd.a04.relation.relation_key_extract.algor.predict.relation_key_pre import extract_keywords

config = Config()


def request(url, data):
    req = requests.post(url, data)
    return json.loads(req.text)


def concat_svo(events):
    svos = []
    for e in events:
        verb = e.get('verb')
        if not verb:
            continue
        subject = e.get('subject')
        obj = e.get('object')
        svo = ''.join([subject, verb, obj])
        svos.append(svo)
    return svos


def extract(graph_db, content, content_id):
    """
    关系抽取
    1、对指代消解后的文章进行分句
    2、调用事件抽取
    3、如果一个句子抽出2个事件，进行关键字抽取，否则流程结束
    4、如果有关键词，调用句子拆分接口，如果无法拆分句子，流程结束
    4、如果拆分出句子，对左句右句分别调用事件抽取
    5、调用关系抽取，将抽取结果不为0的数据分别存入rdms数据库以及图数据库
    :param graph_db:图数据库链接
    :param content:指代消解后的文章
    :param content_id:文章id
    :return: 成功/失败
    """
    try:
        # 去除空格
        content = content.replace(' ', '').replace('\t', '').replace('\n', '').replace(u'\u3000', u'')
        # 分句
        sentences = get_sentences(content)
        # 循环每个句子
        for sentence in sentences:
            # 调用事件抽取
            req_data = {'content': sentence, 'content_id': content_id}
            resp = request(config.url_event_extract, req_data)
            if len(resp.get('event_id_list')) >= 2:
                # 1、调用提取关键词接口
                keywords = extract_keywords(sentence)
                logger.info(f'keywords,{keywords}')
                # 单个关键词，如“导致”
                single = keywords['single']
                # 多个关键词，如“因为-所以”
                multi1 = keywords['multi1']
                multi2 = keywords['multi2']
                if single:
                    keywords = [[w] for w in single]
                elif multi1 and multi2:
                    # 如果两个值有一个为空，keywords就为空
                    keywords = list(product(multi1, multi2))
                else:
                    continue

                splits = []

                for k in keywords:
                    # 2.调用拆分子句
                    split_s = split(sentence, k)
                    if not split_s:
                        continue
                    splits.extend(split_s)
                # 如果没有拆分出句子,结束
                if not splits:
                    continue
                # 拆分句子的接口会返回多种拆分情况，所以这里需要循环
                for s in splits:
                    # 左句
                    left_sentence = s[0]
                    # 右句
                    right_sentence = s[1]
                    left_event_list = []
                    right_event_list = []
                    for event_id in resp.get('event_id_list'):
                        # 查询事件信息
                        event_info = get_event_info(event_id)
                        pattern = rf"{event_info.values[1]}"
                        left_rs = re.search(pattern, left_sentence)
                        right_rs = re.search(pattern, right_sentence)
                        if left_rs is not None:
                            event_dict = {"event_id": event_id, "shorten_sentence": event_info.values[0],
                                          "triggerloc_index": event_info.values[2]}
                            left_event_list.append(event_dict)
                        if right_rs is not None:
                            event_dict = {"event_id": event_id, "shorten_sentence": event_info.values[0],
                                          "triggerloc_index": event_info.values[2]}
                            right_event_list.append(event_dict)
                    if left_event_list and right_event_list:
                        rst = []
                        event_all_list = list(product(left_event_list, right_event_list))
                        for event in event_all_list:
                            classify_resq = int(
                                class_pre(sentence, event[0]["triggerloc_index"], event[1]["triggerloc_index"]))
                            if classify_resq != 0:
                                p = [[event[0]["shorten_sentence"], event[1]["shorten_sentence"]]]
                                h = [[event[0]["event_id"], event[1]["event_id"]]]
                                rst.append({'event_pair': p, 'event_id_pair': h, 'relation': classify_resq})
                        # 关系保存数据库
                        # insert_event_relations(keywords, s[0], s[1], event_pairs, rst, 4, content_id)
                        # 关系保存图数据库
                        logger.info("保存图数据库开始")
                        insert_graph_db(graph_db, rst)
                        logger.info("保存图数据库结束")
                    else:
                        continue
                # 关系抽取

                # rst = []

                # for p, h in zip(event_pairs, event_id_pairs):
                #     # 调用关系抽取
                #     classify_resq = int(class_pre(p[0], p[1]))
                #     if classify_resq != 0:
                #         rst.append({'event_pair': p, 'event_id_pair': h, 'relation': classify_resq})
                #
                # # 关系保存数据库
                # insert_event_relations(keywords, s[0], s[1], event_pairs, rst, 4, content_id)
                # # 关系保存图数据库
                # logger.info("保存图数据库开始")
                # insert_graph_db(graph_db, rst)
                # logger.info("保存图数据库结束")
            else:
                continue
    except Exception as e:
        logger.error(f"{e}")
        return {'status': 'error'}
    # 修改文章状态
    # update(content_id)
    return {'status': 'success'}
