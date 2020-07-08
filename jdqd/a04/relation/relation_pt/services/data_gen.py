from jdqd.a04.relation.relation_pt.algor import relation_combine, relation_util
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
import os
import re
import json
import requests
import glob
from tqdm import tqdm
from feedwork.utils import FileHelper
from jdqd.a04.relation.relation_pt.algor import r_then, r_parallel, r_further


def zh_ratio(text):
    """
    计算中文字符在文本中所占的比例
    :param text:
    :return:
    """
    pattern = re.compile(u'[\u4e00-\u9fa5]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def remove_white_space(text):
    return text.replace(' ', '').replace('\n', '').replace('\t', '')


def remove_html_tags(text):
    # remove comments
    comments_pattern = '<!--.*?-->'
    text = re.sub(comments_pattern, '', text)
    # remove img tag
    tags_pattern = '<imgsrc.*?/>'
    text = re.sub(tags_pattern, '', text)
    return text


def get_articles_from_db():
    import time
    db = DatabaseWrapper('mng')

    sql = 'select article_id, content from t_article_msg'

    t1 = time.time()
    rst = db.query(sql, result_type=QueryResultType.DB_NATURE)
    t2 = time.time()
    print(f'finished query, used {t2 - t1} secs')
    return rst


def get_articles():
    txt = 'articles.txt'
    if os.path.exists(txt):
        articles_zh = json.loads(readfile(txt))
        return articles_zh

    articles = get_articles_from_db()
    articles_zh = filter_db_articles(articles)
    save_content(json.dumps(articles_zh), txt)
    return articles_zh


def save_content(content, fn, mode='w'):
    with open(fn, mode, encoding='utf-8') as f:
        f.write(content)


def readfile(fn, readlines=False):
    with open(fn, 'r', encoding='utf-8') as f:
        if readlines:
            return f.readlines()
        else:
            return f.read()


event_url = 'http://172.168.0.23:38082/event_extract'


def get_events(sentence):
    """
    获取句中的事件, 每个事件按照主语, 否定词, 谓语, 宾语的顺序以竖线|连接成str, 多个事件的
    str以汉字丨(音gun)连接, 返回连接的str
    :param sentence:
    :return:
    """
    print('processing request')
    data = {'sentence': sentence}
    ret = requests.post(event_url, data=data)
    print('finished request', ret.status_code)
    rst = json.loads(ret.text)
    events = rst['data'][0]['events']
    svos = [[e['subject'], e['negative_word'], e['verb'], e['object']] for e in
            events]
    svos = ['|'.join(svo) for svo in svos]
    svos = '丨'.join(svos)
    return svos


def filter_article(content):
    content = remove_white_space(content)
    content = remove_html_tags(content)
    return content


def filter_db_articles(articles):
    """"""
    filtered = []
    for atc in articles:
        id_ = atc['article_id']
        content = atc['content']
        if content is None or len(content) < 4:
            continue
        content = filter_article(content)
        # 如果文章内容全为被remove的内容, 则此时长度有可能为0, 计算语言比例时会
        # divide by zero, 故再次长度判断
        if len(content) < 4:
            continue
        if zh_ratio(content) > .75:
            filtered.append([id_, content])
    return filtered


def extract_article_relation(article_content, relation, neg_num, neg_capacity):
    """
    从文章的句子中抽取关系, 根据是否有关系创建关联词模型的正负样本.
    :param article_content:
    :param relation:
    :param neg_num: 用于记录负样本数量
    :param neg_capacity:
    :return:
    """
    rsts_pos = []
    rsts_neg = []
    sentences = relation_util.split_article_to_sentences(article_content)
    for sentence in sentences:
        rst = relation_combine.extract_all_rules(sentence, relation.rules,
                                                 relation.keyword_rules)
        if rst:
            left = rst['left']
            right = rst['right']
            if (not left) or (not right):
                continue
            keyword = json.dumps(rst['tag_indexes'])
            line = [sentence, keyword, left, right]
            line = '\t'.join(line) + '\n'
            rsts_pos.append(line)
        else:
            if neg_num < neg_capacity:
                rsts_neg.append(sentence + '\n')
                neg_num += 1
    return rsts_pos, rsts_neg, neg_num


def extract_articles_relation(articles_dir, relation, neg_capacity=10000):
    """
    遍历本地磁盘上的文章内容生成关键词模型训练数据,
    :param articles_dir:
    :param relation:
    :param neg_capacity:
    :return:
    """
    rst_articles_pos = []
    rst_articles_neg = []
    articles_fn = os.listdir(articles_dir)
    neg_num = 0
    for fn in tqdm(articles_fn):
        fp = os.path.join(articles_dir, fn)
        with open(fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 2:
            continue
        lines = lines[1:]
        content = ''.join(lines)
        content = remove_white_space(content)
        rsts, rsts_neg, neg_num = extract_article_relation(content, relation,
                                                           neg_num,
                                                           neg_capacity)
        rst_articles_pos.extend(rsts)
        rst_articles_neg.extend(rsts_neg)
    return rst_articles_pos, rst_articles_neg


def save_ner_data(rst_articles_pos, rst_articles_neg, neg_upper_bound,
                  ner_data_pos_fp, ner_data_neg_fp):
    """
    遍历文章
    :param relation:
    :param neg_upper_bound:
    :param ner_data_pos_fp:
    :param ner_data_neg_fp:
    :return:
    """
    save_content(''.join(rst_articles_pos), ner_data_pos_fp)
    save_content(''.join(rst_articles_neg[:neg_upper_bound]), ner_data_neg_fp)


def gen_classify_data(rst_articles_pos, rst_articles_neg, neg_upper_bound,
                      classify_data_pos_fp, classify_data_neg_fp):
    rst_articles_pos = [r.replace('\n', '').split('\t') for r in
                        rst_articles_pos]
    rst_articles_neg = [r.replace('\n', '').split('\t') for r in
                        rst_articles_neg]
    with open(classify_data_pos_fp, 'a', encoding='utf-8') as fcp:
        for rp in rst_articles_pos:
            left = rp[2]
            right = rp[3]
            try:
                left_events = get_events(left)
                right_events = get_events(right)
                line = rp + [left_events, right_events]
                line = '\t'.join(line) + '\n'
                fcp.write(line)
            except Exception as e:
                print(e)
    with open(classify_data_neg_fp, 'a', encoding='utf-8') as fcn:
        for i, rp in enumerate(rst_articles_neg):
            if i > neg_upper_bound:
                continue
            try:
                events = get_events(rp[0])
                line = rp + [events]
                line = '\t'.join(line) + '\n'
                fcn.write(line)
            except Exception as e:
                print(e)



def gen_neg_by_other(relation):
    contrast_neg_f = r'C:\work1\qingdao\dev\python\jdqd\a04\relation\relation_pt\services\contrast_neg.txt'
    new_lines = []
    with open(contrast_neg_f, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split('\t')
            sentence = splits[0]
            is_parallel = bool(
                relation_combine.extract_all_rules(sentence, relation.rules,
                                                   relation.keyword_rules))
            if not is_parallel:
                new_lines.append(line)
    save_content(''.join(new_lines), 'data_parallel_e_neg.txt')


def filter_events():
    new_lines = []
    with open(
        r'C:\work1\qingdao\dev\python\jdqd\a04\relation\relation_pt\services\data_parallel_e.txt',
        'r', encoding='utf-8') as f:
        fs = list(set(f.readlines()))
        for line in fs:
            conts = line.strip().split('\t')
            if len(conts) == 6 and '丨' not in line:
                left_event = conts[4]
                right_event = conts[5]
                if len(left_event) > 3 and len(right_event) > 3:
                    new_line = '\t'.join(
                        [conts[0], left_event, right_event]) + '\n'
                    new_lines.append(new_line)
    save_content(''.join(new_lines), 'data_parallel_e_filter.txt')


if __name__ == '__main__':
    ner_data_fn = 'data_parallel.txt'
    ner_data_neg_fn = 'data_parallel_neg.txt'
    new_ner_data_fn = 'data_parallel_e.txt'
    # add_pos_events(new_ner_data_fn)
    # gen_neg_by_other(r_parallel)
    filter_events()
    pass

    # for id_, content in articles_zh[:5]:
    #     rsts = extract_articles(content, relation)
    #     rst_articles.extend(rsts)
    # save_content(''.join(rst_articles), f'data_{relation.__name__}.txt')
