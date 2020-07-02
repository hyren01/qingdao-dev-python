from jdqd.a04.relation.relation_pt.algor import relation_combine, relation_util
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
import os
import re
import json
import requests
from feedwork.utils import FileHelper
from jdqd.a04.relation.relation_pt.algor import r_then, r_causality, r_contrast

FileHelper.write('test_feedwork_file.txt', 'bbb')


def extract_articles(article_content, relation):
    rsts = []
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
            left_events = get_events(left)
            right_events = get_events(right)
            line = [sentence, str(keyword), left, right, left_events,
                    right_events]
            line = '\t'.join(line) + '\n'
            rsts.append(line)
    return rsts


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


if __name__ == '__main__':
    rst_articles = []
    articles_zh = get_articles()
    relation = r_contrast
    # 格式: 原句\t{'keyword':[start, end]}\t左句\t右句\t左句事件列表\t右句事件列表
    # 其中, 左右句事件列表由事件组成, 以汉字丨(音gun)分隔, 每个事件包含主语负词谓语宾语,
    # 以|分隔
    for id_, content in articles_zh:
        rsts = extract_articles(content, relation)
        rst_articles.extend(rsts)
    save_content(''.join(rst_articles), f'data_{relation.__name__}.txt')
