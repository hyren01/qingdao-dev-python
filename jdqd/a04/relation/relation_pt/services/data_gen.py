from jdqd.a04.relation.relation_pt.algor import relation_combine, relation_util
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
import re
from feedwork.utils import FileHelper
from jdqd.a04.relation.relation_pt.algor import r_then, r_causality

FileHelper.write('test_feedwork_file.txt', 'bbb')


def extract_articles(article_content, relation):
    rsts = []
    sentences = relation_util.split_article_to_sentences(article_content)
    for sentence in sentences:
        rst = relation_combine.extract_all_rules(sentence, relation.rules,
                                                 relation.keyword_rules)
        if rst:
            rsts.append(rst)
    return rsts


def kr_ratio(text):
    pattern = re.compile(u"[\uac00-\ud7ff]")
    matched = pattern.findall(text)
    return len(matched) / len(text)


def jp_ratio(text):
    pattern = re.compile(u"[\u30a0-\u30ff]")
    matched = pattern.findall(text)
    return len(matched) / len(text)


def zh_ratio(text):
    pattern = re.compile(u'[\u4e00-\u9fa5]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def en_ratio(text):
    pattern = re.compile('[a-zA-Z]')
    matched = pattern.findall(text)
    return len(matched) / len(text)


def ru_ratio(text):
    pattern = re.compile(u'[\u0400-\u04ff]')
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


def get_articles():
    import time
    db = DatabaseWrapper('mng')

    sql = 'select article_id, content from t_article_msg'

    t1 = time.time()
    rst = db.query(sql, result_type=QueryResultType.DB_NATURE)
    t2 = time.time()
    print(f'finished query, used {t2 - t1} secs')
    return rst


def filter_articles(articles):
    filtered = []
    for atc in articles:
        id_ = atc['article_id']
        content = atc['content']
        if content is None or len(content) < 4:
            continue
        content = remove_html_tags(content)
        content = remove_white_space(content)
        if zh_ratio(content) > .75:
            filtered.append([id_, content])
    return filtered


if __name__ == '__main__':
    articles = get_articles()
    articles_zh = filter_articles(articles)
    rst_articles = []
    for id_, content in articles_zh:
        rsts = extract_articles(content, r_causality)
        rst_articles.append(rsts)
