import requests
import json
from jdqd.a04.relation.services.common.rdms_db_util import insert_event_relations
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
    2、调用提取关键词接口，如果没有关键词，直接调用事件抽取
    3、如果有关键词，调用句子拆分接口，如果无法拆分句子，直接调用事件抽取
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
            # 1、调用提取关键词接口
            keywords = extract_keywords(sentence)
            # req_keywords_data = {'sentence': sentence}
            # keywords = request(config.url_relation_keywords, req_keywords_data)
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
                # 直接调用事件抽取
                req_data = {'content': sentence, 'content_id': content_id}
                request(config.url_event_extract, req_data)
                continue

            splits = []

            for k in keywords:
                # 2.调用拆分子句
                split_s = split(sentence, k)
                # req_split_data = {'sentence': sentence, 'keyword': json.dumps(k, ensure_ascii=False)}
                # split = request(config.url_relation_split, req_split_data)
                if not split_s:
                    continue
                splits.extend(split_s)
            # 如果没有拆分出句子，在rdms数据库里保存一条数据，再调用事件抽取
            if not splits:
                insert_event_relations(keywords, [], [], [], [], 2, content_id)
                # 原句调用事件抽取
                req_data = {'content': sentence, 'content_id': content_id}
                request(config.url_event_extract, req_data)
                continue
            # 拆分句子的接口会返回多种拆分情况，所以这里需要循环
            for s in splits:
                # 左句
                left_sentence = s[0]
                # 右句
                right_sentence = s[1]
                # 调用事件抽取
                req_left_data = {'content': left_sentence, 'content_id': content_id}
                req_right_data = {'content': right_sentence, 'content_id': content_id}
                left_resp = request(config.url_event_extract, req_left_data)
                # 左句右句调用事件抽取如果其中一个为空，则跳过
                if not left_resp.get('event_id_list'):
                    continue
                right_resp = request(config.url_event_extract, req_right_data)
                if not right_resp.get('event_id_list'):
                    continue
                logger.info(f'svos,left_event:{left_resp["event_id_list"]},right_event:{right_resp["event_id_list"]}')
                # 组成需要进行关系匹配的事件短语
                event_id_pairs = list(product(left_resp["event_id_list"], right_resp["event_id_list"]))
                event_pairs = list(product(left_resp["event_list"], right_resp["event_list"]))
                logger.info(f'event_pairs,{event_pairs}')

                rst = []

                for p, h in zip(event_pairs, event_id_pairs):
                    # 调用关系抽取
                    classify_resq = int(class_pre(p[0], p[1]))
                    # req_classify_data = {'event1': p[0], 'event2': p[1], 'event_id1': h[0], 'event_id2': h[1]}
                    # classify_resq = request(config.url_relation_classify, req_classify_data)
                    if classify_resq != 0:
                        rst.append({'event_pair': p, 'event_id_pair': h, 'relation': classify_resq})

                # 关系保存数据库
                insert_event_relations(keywords, s[0], s[1], event_pairs, rst, 4, content_id)
                # 关系保存图数据库
                logger.info("保存图数据库开始")
                insert_graph_db(graph_db, rst)
                logger.info("保存图数据库结束")
    except Exception as e:
        logger.error(f"{e}")
        return {'status': 'error'}
    # 修改文章状态
    # update(content_id)
    return {'status': 'success'}
