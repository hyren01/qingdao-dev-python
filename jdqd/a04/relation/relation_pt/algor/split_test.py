import psycopg2
import json
import requests


def get_conn():
    database = "ebmdb2"
    user = "jdqd"
    password = "jdqd"
    host = "139.9.126.19"
    port = "31001"
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    print('connection ok')
    return conn


def query(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return results


def get_sentece_by_id(sid):
    conn = get_conn()
    sql = f"select event_sentence from ebm_event_sentence_copy1 where sentence_id = '{sid}'"
    sentence = query(conn, sql)
    return sentence


def request(url, data):
    req = requests.post(url, data)
    return json.loads(req.text)


from jdqd.a04.relation.relation_pt.algor import r_causality
from jdqd.a04.relation.relation_pt.algor import r_assumption
from jdqd.a04.relation.relation_pt.algor import r_condition
import jdqd.a04.relation.relation_pt.algor.relation_util

import split_server

relations = [r_causality, r_assumption, r_condition]

# sentence = '例如，教师的工作量减少，因为学生越来越少，越来越多的校舍被遗弃，教师非法地忽视了学生越来越少和校舍越来越多的责任，在日益增长的市场经济中工作'
sentence = '我在三池渊看到的三胞胎是那些为了我在三池渊看到的三胞胎而绞尽脑汁的人当我在三池渊看到的三胞胎对姐妹们说当我在三池渊看到的三胞胎遇到了一个好的配偶并且成家的时候我在三池渊看到的三胞胎一定要给我在三池渊看到的三胞胎写信'
# keyword = ['只有', '才']
# keyword = ['由于']
keyword = ['是为了']
min_sentences, delimiters = relation_util.split_sentence(sentence)
keyword_pos = relation_util.get_keyword_pos(min_sentences, keyword[0])

rst, source = split_server.split(sentence, keyword)

rst_split, rule = split_server.split_by_relation(r_causality, sentence, keyword, keyword_pos)

import time

t = time.time()
conn = get_conn()
sql = 'select event_relations.sentence_id, event_sentence, words, left_sentence ' \
      'from event_relations left join ebm_event_sentence_copy1 ' \
      'on event_relations.sentence_id=ebm_event_sentence_copy1.sentence_id'
rst = query(conn, sql)

rst = [[i, s, json.loads(w), json.loads(__)] for i, s, w, __ in rst]

for sid, sentence, w, __ in rst:
    # if sid != '4870c1dc80a511ea80070242ac12000a':
    #     continue
    if not sentence:
        continue
    for w_ in w:
        if len(w_) == 2:
            continue
        print('-' * 36)
        rst, source = split_server.split(sentence, w_)
        print(w_)
        print(sentence)
        print(rst)
        print(source)

'''
url_relation_split = "http://localhost:5000/relation_split"
for sid, w, s in rst:
    # if sid != 'dc5ae87e80a911ea9ba90242ac12000a':
    #     continue
    sentence = get_sentece_by_id(sid)
    if not sentence:
        continue

    sentence = sentence[0][0]
    print('-' * 36)
    for w_ in w:
        req_split_data = {'sentence': sentence, 'keyword': json.dumps(w_)}

        split_rst = request(url_relation_split, req_split_data)
        print(sentence)
        print(s)
        print(split_rst)
        # if split_rst is None:
        #     print(sid)
        #     break
'''
