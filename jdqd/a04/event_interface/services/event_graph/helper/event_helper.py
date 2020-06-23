#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging
import jdqd.a04.event_interface.services.event_graph.helper.graph_database as graph_service
import jdqd.a04.event_interface.services.event_graph.helper.rdms_database as rdms_service

from jdqd.a04.event_interface.config.project import Config
from jdqd.a04.event_interface.services.common.http_util import http_post

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")
config = Config()
global_event_tag = config.global_event_tag
synonyms = None  # 同义词库


def __date_formatter(date_str):
    if date_str is None or date_str == '':
        return ''
    date_str = str.strip(date_str)
    date_str = date_str.replace("号", "日")
    date_str = date_str.replace("月份", "月")
    return date_str


def __init_and_get_synonyms(rdms_db, update_synonyms):
    """
    根据主语、谓语、宾语对事件表进行字符匹配。注意，此方法会更新synonyms全局变量。

    :param rdms_db: object.RDMS数据库连接对象
    :param update_synonyms: boolean.是否更新同义词
    """
    global synonyms
    if synonyms is None or update_synonyms:
        synonyms = {}
        synonym_info = rdms_service.get_synonym(rdms_db)
        for synonym in synonym_info:
            source_word = str(synonym['source_word'])
            targer_word = str(synonym['targer_word'])
            # 在同义词中，A与B同义，这意味着A的同义词是B、B的同义词是A。
            # synonyms的结构是{"特朗普": ['川普', '唐纳德']}
            if source_word in synonyms:
                synonyms[source_word].append(targer_word)
            else:
                synonyms[source_word] = [targer_word]
            if targer_word in synonyms:
                synonyms[targer_word].append(source_word)
            else:
                synonyms[targer_word] = [source_word]
        # 对同义词二次归并，如：数据库中[{'source':'中美两国', 'target':'中国和美国'}, {'source':'中美两国': 'target':'美国与中国'}]
        # synonyms中为{'中美两国':['中国和美国', '美国与中国'], '中国和美国':['中美两国'], '美国与中国':['中美两国']}
        # 执行以下代码后，synonyms中为{'中美两国':['中国和美国', '美国与中国'], '中国和美国':['中美两国', '美国与中国'], '美国与中国':['中美两国', '美国与中国']}
        for key in synonyms:
            synonym_word_list = synonyms[key]
            for synonym_word in synonym_word_list:
                if synonym_word in synonyms:
                    synonyms[key] = list(set(synonym_word_list.__add__(synonyms[synonym_word])))
        logging.info("同义词加载成功")

    return synonyms


def __check_event_info_by_char(rdms_db, subject, verb, object, event_negaword, update_synonyms):
    """
    根据主语、谓语、宾语对事件表进行字符匹配。

    :param rdms_db: object.RDMS数据库连接对象
    :param subject: string.主语
    :param verb: string.谓语
    :param object: string.宾语
    :param event_negaword: string.否定词
    :param update_synonyms: boolean.是否更新同义词
    :return 字符匹配事件是否成功。若匹配成功则返回：True, 事件列表；若匹配失败则返回：False, None。
    """
    synonyms = __init_and_get_synonyms(rdms_db, update_synonyms)
    subject_check = synonyms[subject].__add__([subject]) if subject in synonyms else [subject]
    verb_check = synonyms[verb].__add__([verb]) if verb in synonyms else [verb]
    object_check = synonyms[object].__add__([object]) if object in synonyms else [object]
    # 字符匹配事件表
    event_infos = rdms_service.get_event_info(rdms_db, subject_check, verb_check, object_check, event_negaword)
    if event_infos is not None and len(event_infos) > 0:
        return True, event_infos
    else:
        return False, None


def __check_event_attribute_by_char(rdms_db, event_id, event_date, event_local, nentity_org, nentity_person):
    """
    根据事件发生日期、事件发生地点、组织机构、人物对属性表进行字符匹配。

    :param rdms_db: object.RDMS数据库连接对象
    :param event_id: string.事件id
    :param event_date: string.事件发生日期
    :param event_local: string.事件发生地点
    :param nentity_org: string.组织机构
    :param nentity_person: string.人物
    :return 字符匹配事件属性是否成功。若匹配成功则返回：True, 事件属性列表；若匹配失败则返回：False, None。
    """
    event_date = __date_formatter(event_date)
    ebm_event_attributes = rdms_service.get_event_attribute(rdms_db, event_id, event_local, event_date, nentity_org,
                                                            nentity_person)
    if ebm_event_attributes is not None and len(ebm_event_attributes) > 0:
        return True, ebm_event_attributes
    else:
        return False, None


def __check_event_attribute_by_nn(event_local, event_date, nentity_org, nentity_person):
    """
    NN网络属性匹配。NN网络属性匹配为空实现。

    :param event_date: string.事件发生日期
    :param event_local: string.事件发生地点
    :param nentity_org: string.组织机构
    :param nentity_person: string.人物
    :return NN匹配事件属性是否成功。
    """
    return False


def __check_event_info_by_nn(short_sentence, cameo):
    """
    NN网络事件匹配。

    :param short_sentence: string.事件短语
    :param cameo: string.CAMEO CODE
    :return NN匹配事件是否成功。若匹配成功则返回：True, 事件列表；若匹配失败则返回：False, None。
    """
    data = {"short_sentence": short_sentence, "cameo": cameo, "threshold": config.event_similarly_threshold1}
    res = http_post(data, config.event_similarly_uri)
    response = json.loads(res)
    if response["status"] != 'success':
        logging.warning("调用事件相似度匹配接口失败，该事件跳过：" + short_sentence)
        return False, None
    result = response["result"]
    if result is None or len(result) < 1:
        return False, None
    else:
        return True, result


def __parsed_sentence_and_insert2db(rdms_db, graph_db, fulltext_db, content_id, content, event_tag, update_synonyms):
    """
    对传入的句子做事件抽取、组成成份分析、事件类型分析等，以及将分析结果入库。
    :param rdms_db: object.关系数据库连接对象。
    :param graph_db: object.图数据库连接对象。
    :param fulltext_db: object.全文检索服务连接对象。
    :param content_id: String.文本id。
    :param content: String.文本。
    :param event_tag: String.事件节点标签（默认Event）。
    :return content_id, event_id。content_id：若传入的content_id为空，则程序会生成一个id并返回；event_id：事件id
    """
    # 6、程序调用“事件抽取接口”，接口根据事件短句进行分析，接口返回事件短句对应的主谓宾、命名实体、事件发生日期、事件发生地点、事件否定词、事件发生状态、情感分析、CAMEO CODE；
    # content_id = ""
    # event_id = ""
    event_id_list = []
    event_list = []
    data = {"sentence": content}
    res = http_post(data, config.event_extract_uri)
    response = json.loads(res)
    # data = {"sentence": content, "sentence_id": content_id}
    # res = http_post(data, config.relextract_interface_uri)
    # response = json.loads(res)
    sentence_parsed_array = response["data"]
    # 接口返回句子及其事件（主谓宾）
    for sentence_parsed in sentence_parsed_array:
        zh_sentence = sentence_parsed["sentence"]
        content_id, sentence_id = rdms_service.insert_event_sentence(rdms_db, content_id, zh_sentence)
        # 程序调用“关系抽取接口”，获取下标，与事件抽取的下标对比，存图数据库；
        # relextract_data = {"sentence": zh_sentence}
        # relextract_res = http_post(relextract_data, config.relextract_MM_interface_uri)
        # relextract_response = json.loads(relextract_res)
        # if not relextract_response:
        for event in sentence_parsed["events"]:
            if not event["cameo"]:
                continue
            subject = event["subject"]
            verb = event["verb"]
            object = event["object"]
            namedentity = event["namedentity"]
            namedentity_location = namedentity["location"]
            namedentity_organization = namedentity["organization"]
            namedentity_person = namedentity["person"]
            # sentiment_analysis = sentence_parsed["sentiment_analysis"]
            sentiment_analysis = ""
            event_datetime = event["event_datetime"]
            event_location = event["event_location"]
            negative_word = event["negative_word"]
            state = event["state"]
            cameo = event["cameo"]
            triggerloc_index = event["triggerloc_index"]
            short_sentence = subject + negative_word + verb + object
            subject = ",".join(subject) if type(subject) == list else subject
            verb = ",".join(verb) if type(verb) == list else verb
            object = ",".join(object) if type(object) == list else object
            # 根据主语、谓语、宾语、否定词匹配事件表（字符匹配）
            check_event_info, event_infos = __check_event_info_by_char(rdms_db, subject, verb, object, negative_word,
                                                                       update_synonyms)
            namedentity_location_str = ",".join(namedentity_location)
            namedentity_organization_str = ",".join(namedentity_organization)
            namedentity_person_str = ",".join(namedentity_person)
            # 事件匹配成功，则进行匹配属性表（字符匹配）
            if check_event_info:
                for event_info in event_infos:
                    event_id_list.append(event_info['event_id'])
                    event_list.append(event_info['shorten_sentence'])
                    event_id = event_info['event_id']
                    logging.info("字符匹配事件成功，事件id为：" + event_id)
                    check_event_attribute, ebm_event_attributes = __check_event_attribute_by_char(
                        rdms_db, event_id, event_datetime, event_location,
                        namedentity_organization, namedentity_person)
                    # 若字符属性匹配成功，意味着该事件已经存在，仅仅记录句子与这些属性的关系
                    if check_event_attribute:
                        for event_attribute in ebm_event_attributes:
                            attribute_id = event_attribute['relation_id']
                            rdms_service.insert_sentattribute_rel(rdms_db, short_sentence, attribute_id, sentence_id)
                    else:
                        # 若字符属性匹配失败，则认为该事件为新事件，记录到事件属性表、事件句子属性关系表
                        attribute_id = rdms_service.insert_event_attribute(rdms_db, sentiment_analysis,
                                                                           event_datetime, event_location, state,
                                                                           namedentity_location_str,
                                                                           namedentity_organization_str,
                                                                           namedentity_person_str, "", event_id)
                        rdms_service.insert_sentattribute_rel(rdms_db, short_sentence, attribute_id, sentence_id)
                        # 若字符属性匹配失败，则进行NN网络属性匹配（NN网络属性匹配为空实现）
                        check_event_attribute = __check_event_attribute_by_nn(event_location, event_datetime,
                                                                              namedentity_organization,
                                                                              namedentity_person)
                        # 若NN属性匹配成功，则记录到事件属性校验表（目前无实现）
                        if check_event_attribute:
                            pass
                        else:
                            pass
            else:
                # 字符事件匹配失败，则记录事件信息表、事件属性表、事件句子属性表、图数据库
                event_id, event = __storage_new_event(rdms_db, subject, verb, object, short_sentence, cameo,
                                                      triggerloc_index,
                                                      sentiment_analysis, event_datetime, negative_word, state,
                                                      event_location,
                                                      namedentity_location_str, namedentity_organization_str,
                                                      namedentity_person_str, graph_db, sentence_id, event_tag)
                event_list.append(event)
                event_id_list.append(event_id)
                # 字符事件匹配失败，则进行NN事件匹配
                check_event_info, event_infos = __check_event_info_by_nn(short_sentence, cameo)
                # 若NN事件匹配成功，记录到事件信息镜像校验表
                if check_event_info:
                    copy_event_id = event_id
                    for event_info in event_infos:
                        event_id = event_info["event_id"]
                        rdms_service.insert_event_copy(rdms_db, copy_event_id, event_id)
                else:
                    # 若NN事件匹配失败，则保存神经网络向量
                    data = {"short_sentence": short_sentence, "cameo": cameo, "event_id": event_id}
                    res = http_post(data, config.event_vector_uri)
                    response = json.loads(res)
                    if response["status"] != 'success':
                        logging.warning("调用事件向量存储接口失败：" + response["message"])
    # 上面代码中出错后不需要回滚
    rdms_db.commit()

    return event_id_list, event_list


def __storage_new_event(rdms_db, subject, verb, object, short_sentence, cameo, triggerloc_index, sentiment_analysis,
                        event_datetime, negative_word, state, event_local, event_location, event_organization,
                        event_person, graph_db, sentence_id, event_tag):
    # 11、接口生成事件id，并同主语、谓语、宾语、主谓宾构成的事件短语以及CAMEO编号存入“事件信息表”；
    # 接口生成属性id、事件生成日期，并同情感分析、事件发生日期、事件发生地点、事件否定词、事件发生状态、
    # 命名实体、句子id、事件id存入“事件与事件句子关系表”（事件属性表）；
    # 将属性id（事件属性表）、句子id（事件句子表）、存入“事件句子属性关系表”。
    event_id = rdms_service.insert_event_info(rdms_db, subject, verb, object, short_sentence, cameo,
                                              triggerloc_index, negative_word)
    event_datetime = __date_formatter(event_datetime)
    attribute_id = rdms_service.insert_event_attribute(rdms_db, sentiment_analysis, event_datetime, event_local, state,
                                                       event_location, event_organization, event_person, "", event_id)
    rdms_service.insert_sentattribute_rel(rdms_db, short_sentence, attribute_id, sentence_id)

    # 12、程序将数据存储于图数据库，接口标识图中的事件节点为“Event”，并将事件id、事件短语、
    # 事件发生日期属性存储为事件节点，接口根据关系抽取出的实际关系标识图中的关系节点，并将关系中文名
    # 属性存储为关系节点；
    success = graph_service.create_event(graph_db, event_id, short_sentence, event_datetime, event_tag)
    if success is not True:
        logging.warning("在图数据库中创建事件节点失败")
    return event_id, short_sentence


def get_events_by_search_helper(search, event_tag, start_date, end_date):
    """
     根据传入的文本进行全文检索匹配，开始结束日期做条件过滤查询。
     :param search: String.待检索文本。
     :param event_tag: String.图数据库中的事件标签，传入空时默认为Event。
     :param start_date: String.开始日期。
     :param end_date: String.结束日期。
     :return 图数据库中的事件节点数组，如：{"status": "success", "events": [{"event_sentence": "", "event_id": "", "event_date": ""}]}
     """
    # 2、程序调用“事件类型分析接口”，根据用户输入的搜索句子，接口返回用户输入句子的CAMEO编号数组；
    data = {"sentence": search}
    res = http_post(data, config.event_extract_uri)
    response = json.loads(res)
    sentence_parsed_array = response["data"]
    if len(sentence_parsed_array) < 1:
        return 'success', []
    cameo_code_list = []
    for sentence_parsed in sentence_parsed_array:
        for event in sentence_parsed["events"]:
            cameo_code = event["cameo"]
            if cameo_code is None or cameo_code == '':
                cameo_code = '000'
            cameo_code_list.append(cameo_code)

    # data = {"sentence": search}
    # res = http_post(data, config.constituency_parsed_uri)
    # res_dict = json.loads(res)
    # constituency_txt = res_dict["constituency"]
    # content = res_dict["content"]
    #
    # data = {"constituency": constituency_txt, "content": content}
    # res = http_post(data, config.event_code_parsed_uri)
    # res_dict = json.loads(res)
    # cameo_code = res_dict["cameo_code"]
    # cameo_code = list(set(cameo_code))
    # cameo_code = ','.join(cameo_code)

    # 3、程序调用“文本相似度匹配接口”得到匹配的事件。并根据每个事件的事件发生日期、CAMEO编号，与输入条件进行匹配过滤，
    # 得到最终符合的事件清单，接口返回事件id数组；
    event_id_list = []
    for cameo_code in cameo_code_list:
        if len(event_id_list) >= config.event_similarly_topN:
            break
        data = {"short_sentence": search, "cameo": cameo_code, "threshold": config.event_similarly_threshold2}
        res = http_post(data, config.event_similarly_uri)
        response = json.loads(res)
        if response["status"] != 'success':
            logging.warning("调用事件相似度匹配接口失败，该事件搜索跳过：" + search)
            return 'error', []
        result = response["result"]
        for item in result:
            event_id_list.append(item["event_id"])
            if len(event_id_list) >= config.event_similarly_topN:
                break

    if len(event_id_list) < 1:
        return 'success', []

    # data = {"search": search, "cameo_code": cameo_code, "start_date": start_date, "end_date": end_date}
    # res = http_post(data, config.fulltext_match_uri)
    # res_dict = json.loads(res)
    # event_id_list = res_dict["event_ids"]

    # 4、程序调用“图数据查询接口”，在图数据库中根据事件id数组查询出对应的事件节点数组；
    if event_tag is None:
        event_tag = global_event_tag
    data = {"event_ids": event_id_list, "event_tag": event_tag, "start_date": start_date, "end_date": end_date}
    res = http_post(data, config.get_events_uri)
    res_dict = json.loads(res)

    return res_dict['status'], res_dict['events']


def event_parsed_extract_helper(rdms_db, graph_db, fulltext_db, content, content_id, event_tag, update_synonyms):
    """
    对传入的文本进行分句，并根据句子中事件的关系进行不同的解析方式。
    :param rdms_db: object.关系数据库连接对象。
    :param graph_db: object.图数据库连接对象。
    :param fulltext_db: object.全文检索服务连接对象。
    :param content_id: String.文本id。
    :param content: String.中文句子。
    :param event_tag: String.事件节点标签（默认Event）。
    :param update_synonyms: boolean.是否更新同义词。
    :return boolean, content_id。boolean：Boolean类型，表示解析是否成功；content_id：若传入的content_id为空，则程序会生成一个id并返回。
    """
    # 2、程序将原始语种字符串翻译为英文；
    # content = translate_any_2_anyone(content, "en")
    # if content == '':
    #     return {"status": "error"}
    # # 3、程序调用“指代消解接口”，对英文文章进行指代消解，接口返回指代消解后的英文文章；
    # data = {"content": content}
    # res = http_post(data, config.coref_interface_uri)
    # res_dict = json.loads(res)
    # if res_dict['status'] != 'success':
    #     return False, "在进行指代消解时发生异常"
    # content = res_dict["coref"]
    # 将指代消解后的英文文本存入t_article_msg_en表
    # rdms_service.insert_corecontent(rdms_db, content_id, content)
    # 4、程序将英文文章翻译为中文文章，并将文章分句；
    # content = translate_any_2_anyone(content, "zh")
    # zh_sentence_array = re.findall(zhsplite_sentence, coref_content)
    # for zh_sentence in zh_sentence_array:
    # 5、程序调用“关系抽取接口”，接口对中文文章进行关系抽取，接口对应的事件短句组、事件短语间的关系；
    # data = {"sentence": zh_sentence}
    # res = http_post(data, config.relextract_interface_uri)
    # res_dict = json.loads(res)
    # res_dict = []
    # 将指代消解后的中文文本存入t_article_msg_zh表
    # rdms_service.insert_zhcontent(rdms_db, content_id, content)

    # if not res_dict:
    event_id_list, event_list = __parsed_sentence_and_insert2db(rdms_db, graph_db, fulltext_db, content_id, content,
                                                                event_tag, update_synonyms)
    # if not content_id:
    #     continue
    # for relation_item in res_dict:
    #     relation_type = relation_item["relation"]
    #     # relation_type = ''
    #     # 若事件关系抽取程序对句子没有抽取出关系，则
    #     if relation_type == '' or relation_type is None:
    #         content_id, event_id = __parsed_sentence_and_insert2db(rdms_db, graph_db, fulltext_db, content_id,
    #                                                                coref_content, event_tag)
    #     else:
    #         tag = relation_item["tag"]
    #         source = relation_item["source"]
    #         target = relation_item["target"]
    #         # 此处有问题，一个关系短句中可能会抽取出两个事件，暂时认为，每个事件短句最多存在1个事件
    #         # 若一个句子抽取出了一组关系短句，每个关系短句存在两个事件，那么这N个事件间的关系如何组织还未考虑。
    #         content_id, cause_event_id = __parsed_sentence_and_insert2db(rdms_db, graph_db, fulltext_db, content_id,
    #                                                                      source, event_tag)
    #         content_id, effect_event_id = __parsed_sentence_and_insert2db(rdms_db, graph_db, fulltext_db,
    #                                                                       content_id, target, event_tag)
    #
    #         if cause_event_id != '' and cause_event_id is not None and effect_event_id != '' \
    #                 and effect_event_id is not None:
    #             success = graph_service.create_relation(graph_db, tag, cause_event_id, effect_event_id,
    #                                                     relation_type, event_tag)
    #             if success is not True:
    #                 logging.warning("创建事件关系失败")

    # 14、接口返回内容id（文章或者句子的id）。
    return True, event_id_list, event_list


def get_article_list(event_id):
    result_article = rdms_service.get_article_list(event_id)
    return result_article


def get_article_info(article_id, event_id):
    result_article, result_sentence = rdms_service.get_article_info(article_id, event_id)
    return result_article, result_sentence


def get_event_to_event(graph_db, event_tag, event_source_sentence, event_target_sentence):
    result_link, result_data = graph_service.get_event_to_event(graph_db, event_tag, event_source_sentence,
                                                                event_target_sentence)
    return result_link, result_data


if __name__ == '__main__':
    import re


    def get_strtime(date_str):
        date_str = date_str.replace("年", "-").replace("月", "-").replace("日", " ").replace("号", " ") \
            .replace("/", "-").strip()
        regex_list = [
            # 2013年8月15日 22:46:21
            r"(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2})",
            # "2013年8月15日 22:46"
            r"(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2})",
            # "2014年5月11日"
            r"(\d{4}-\d{1,2}-\d{1,2})",
            # "2014年5月"
            r"(\d{4}-\d{1,2})",
            # "2014年"
            r"(\d{4}-)",
            # "9月"
            r"(\d{1,2}-)",
        ]
        for regex in regex_list:
            datetime = re.search(regex, date_str)
            if datetime:
                datetime = datetime.group(1)
                return datetime

        return date_str


    a = "自2013年以来"
    print(get_strtime(a))
