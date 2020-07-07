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


def extract_articles(article_content, relation, neg_num, with_events=True,
                     neg_capacity=10000):
    rsts = []
    rsts_neg = []
    sentences = relation_util.split_article_to_sentences(article_content)
    for sentence in sentences:
        rst = relation_combine.extract_all_rules(sentence, relation.rules,
                                                 relation.keyword_rules)
        if rst:
            left = rst['left']
            right = rst['right']
            keyword = rst['tag_indexes']
            if (not left) or (not right):
                continue
            if with_events:
                left_events = get_events(left)
                right_events = get_events(right)
                line = [sentence, str(keyword), left, right, left_events,
                        right_events]
            else:
                line = [sentence, str(keyword), left, right, '', '']
            line = '\t'.join(line) + '\n'
            rsts.append(line)
        else:
            if neg_num < neg_capacity:
                rsts_neg.append(sentence + '\n')
                neg_num += 1
    return rsts, rsts_neg, neg_num


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
    articles_zh = filter_articles(articles)
    save_content(json.dumps(articles_zh), txt)
    return articles_zh


def save_content(content, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(content)


def readfile(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        return f.read()


event_url = 'http://172.168.0.23:38082/event_extract'


def get_events(sentence):
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


def filter_articles(articles):
    filtered = []
    for atc in articles:
        id_ = atc['article_id']
        content = atc['content']
        if content is None or len(content) < 4:
            continue
        content = remove_white_space(content)
        content = remove_html_tags(content)
        # 如果文章内容全为被remove的内容, 则此时长度有可能为0, 计算语言比例时会
        # divide by zero, 故再次长度判断
        if len(content) < 4:
            continue
        if zh_ratio(content) > .75:
            filtered.append([id_, content])
    return filtered


def save_ner_data():
    rst_articles = []
    rst_articles_neg = []
    # articles_zh = get_articles()
    relation = r_parallel
    # 格式: 原句\t{'keyword':[start, end]}\t左句\t右句\t左句事件列表\t右句事件列表
    # 其中, 左右句事件列表由事件组成, 以汉字丨(音gun)分隔, 每个事件包含主语负词谓语宾语,
    # 以|分隔
    articles_dir = r'C:\work1\qingdao\archive\articles\shizheng2'
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
            rsts, rsts_neg, neg_num = extract_articles(content, relation,
                                                       neg_num, False)
            rst_articles.extend(rsts)
            rst_articles_neg.extend(rsts_neg)
    save_content(''.join(rst_articles), ner_data_fn)
    save_content(''.join(rst_articles_neg[:20000]), ner_data_neg_fn)


def add_pos_events(new_ner_data_fn):
    with open(ner_data_fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(new_ner_data_fn, 'a', encoding='utf-8') as f_w:
            for line in lines:
                splits = line.split('\t')
                left = splits[2]
                right = splits[3]
                try:
                    left_events = get_events(left)
                    right_events = get_events(right)
                    splits[4] = left_events
                    splits[5] = right_events
                    new_line = '\t'.join(splits) + '\n'
                    f_w.write(new_line)
                except Exception as e:
                    print(e)



def add_neg_events(new_ner_data_neg_fn, neg_num, neg_capacity):
    with open(ner_data_neg_fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(new_ner_data_neg_fn, 'a', encoding='utf-8') as f_w:
            for line in lines:
                splits = line.split('\t')
                left = splits[2]
                right = splits[3]
                try:
                    left_events = get_events(left)
                    right_events = get_events(right)
                    splits[4] = left_events
                    splits[5] = right_events
                    new_line = '\t'.join(splits) + '\n'
                    f_w.write(new_line)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    ner_data_fn = 'data_parallel.txt'
    ner_data_neg_fn = 'data_parallel_neg.txt'
    new_ner_data_fn = 'data_parallel_e.txt'
    add_pos_events(new_ner_data_fn)
    pass

    # for id_, content in articles_zh[:5]:
    #     rsts = extract_articles(content, relation)
    #     rst_articles.extend(rsts)
    # save_content(''.join(rst_articles), f'data_{relation.__name__}.txt')
