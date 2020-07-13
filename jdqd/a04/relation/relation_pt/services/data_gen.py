from jdqd.a04.relation.relation_pt.algor import relation_combine, relation_util
from feedwork.database.database_wrapper import DatabaseWrapper
from feedwork.database.enum.query_result_type import QueryResultType
from feedwork.AppinfoConf import ALGOR_PRETRAIN_ROOT
from feedwork.utils.FileHelper import cat_path
import os
import re
import json
import requests
from tqdm import tqdm
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
    return text.replace(' ', '').replace('\n', '').replace('\t', '').replace('\u3000', '')


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
            keyword = rst['tag_indexes']
            line = [sentence, keyword, left, right]
            rsts_pos.append(line)
        else:
            if neg_num < neg_capacity:
                rsts_neg.append(sentence)
                neg_num += 1
    return rsts_pos, rsts_neg, neg_num


def extract_articles_relation(articles_content, relation, neg_capacity):
    """
    遍历文章内容生成关键词模型训练数据,
    :param articles_dir:
    :param relation:
    :param neg_capacity:
    :return:
    """
    rst_articles_pos = []
    rst_articles_neg = []
    neg_num = 0
    for content in tqdm(articles_content):
        content = remove_white_space(content)
        rsts, rsts_neg, neg_num = extract_article_relation(content, relation,
                                                           neg_num,
                                                           neg_capacity)
        rst_articles_pos.extend(rsts)
        rst_articles_neg.extend(rsts_neg)
    return rst_articles_pos, rst_articles_neg


articles_db_path = cat_path(ALGOR_PRETRAIN_ROOT, 'articles.txt')


def get_db_articles():
    """
    获取数据库中爬取的文章内容. 如果有本地的内容文件, 则从本地读取, 否则从数据库读取,
    并将过滤后的结果存到本地文件
    :return:
    """
    if os.path.exists(articles_db_path):
        articles_zh = json.loads(readfile(articles_db_path))
        return articles_zh

    articles = get_articles_from_db()
    articles_zh = filter_db_articles(articles)
    save_content(json.dumps(articles_zh, ensure_ascii=False), articles_db_path)
    return articles_zh


def get_articles_content(articles_dir, from_db=False):
    if from_db:
        articles_content = get_db_articles()
        # 取下标1位置内容. 下标0位置为文章id
        articles_content = [a[1] for a in articles_content]
    else:
        articles_fn = os.listdir(articles_dir)
        articles_content = []
        for fn in tqdm(articles_fn):
            fp = os.path.join(articles_dir, fn)
            with open(fp, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            # 第一行为标题, 且结尾无标点, 不好处理, 直接弃用
            lines = lines[1:]
            content = ''.join(lines)
            articles_content.append(content)
    return articles_content


def save_ner_data(rst_articles_pos, rst_articles_neg, neg_upper_bound,
                  ner_data_pos_fp, ner_data_neg_fp):
    # save pos
    lines_all_pos = []
    for rp in rst_articles_pos:
        sentence = rp[0]
        keyword_indexes = rp[1]
        tag = ['O'] * len(sentence)
        indexes = list(keyword_indexes.values())
        if len(indexes) == 1:
            index1 = indexes[0]
            tag[index1[0]:index1[1]] = ['B-S'] + (index1[1] - index1[0] - 1) * [
                'I-S']

        if len(indexes) == 2:
            index1 = indexes[0]
            tag[index1[0]:index1[1]] = ['B-C'] + (index1[1] - index1[0] - 1) * [
                'I-C']
            index2 = indexes[1]
            tag[index2[0]:index2[1]] = ['B-E'] + (index2[1] - index2[0] - 1) * [
                'I-E']
        lines_sentence = ''.join(
            [f'{c}\t{t}\n' for c, t in zip(sentence, tag)]) + '\n'
        lines_all_pos.append(lines_sentence)
    save_content(''.join(lines_all_pos), ner_data_pos_fp)

    # save neg
    lines_all_neg = []
    for rn in rst_articles_neg[:neg_upper_bound]:
        lines_sentence = ''.join([f'{c}\tO\n' for c in rn]) + '\n'
        lines_all_neg.append(lines_sentence)
    save_content(''.join(lines_all_neg), ner_data_neg_fp)


def save_classify_data(rst_articles_pos, rst_articles_neg, neg_upper_bound,
                       classify_data_pos_fp, classify_data_neg_fp):
    """
    保存用于判定模型的数据, 正例与负例样本分别保存至不同的文件.
    其中, 正例样本保存内容:
        原句, tag_indexes, 左句, 右句, 左句事件1丨左句事件2...  , 右句事件1丨右句事件2...
    负例样本保存内容:
        原句, 事件1丨事件2...
    事件保存内容: 主语|否定词|谓语|宾语

    :param rst_articles_pos:
    :param rst_articles_neg:
    :param neg_upper_bound:
    :param classify_data_pos_fp: 正例数据保存路径. 正例文件命名格式: classify_{relation}_pos.txt
    :param classify_data_neg_fp: 负例数据保存路径. 负例文件命名格式: classify_{relation}_neg.txt
    :return:
    """
    with open(classify_data_pos_fp, 'a', encoding='utf-8') as fcp:
        for rp in rst_articles_pos:
            left = rp[2]
            right = rp[3]
            try:
                left_events = get_events(left)
                right_events = get_events(right)
                line = rp + [left_events, right_events]
                # line[1] 为tag_indexes of type dict, 需转为str
                line[1] = json.dumps(line[1], ensure_ascii=False)
                line = '\t'.join(line) + '\n'
                fcp.write(line)
            except Exception as e:
                print(e)
    with open(classify_data_neg_fp, 'a', encoding='utf-8') as fcn:
        for i, rp in enumerate(rst_articles_neg):
            if i > neg_upper_bound:
                break
            try:
                events = get_events(rp)
                line = [rp, events]
                line = '\t'.join(line) + '\n'
                fcn.write(line)
            except Exception as e:
                print(e)


def gen_neg_by_other(relation):
    """
    使用其他关系已经生成的事件负例, 经本关系过滤后生成本关系所需的负例, 而不用重复调用事件抽取模块
    :param relation:
    :return:
    """
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


def gen_relation(relation, articles_dir, use_db=False):
    articles_content = get_articles_content(articles_dir, use_db)
    rst_articles_pos, rst_articles_neg = extract_articles_relation(
        articles_content, relation, 10000)
    relation_name = relation.__name__.split('.')[-1]
    ner_pos_fp = cat_path(ALGOR_PRETRAIN_ROOT, 'relation_key_extract',
                          f'ner_{relation_name}_pos.txt')
    ner_neg_fp = cat_path(ALGOR_PRETRAIN_ROOT, 'relation_key_extract',
                          f'ner_{relation_name}_neg.txt')
    save_ner_data(rst_articles_pos, rst_articles_neg, 1000, ner_pos_fp,
                  ner_neg_fp)
    classify_pos_fp = cat_path(ALGOR_PRETRAIN_ROOT, 'relation_extract',
                          f'classify_{relation_name}_pos.txt')
    classify_neg_fp = cat_path(ALGOR_PRETRAIN_ROOT, 'relation_extract',
                          f'classify_{relation_name}_neg.txt')
    save_classify_data(rst_articles_pos, rst_articles_neg, 1000, classify_pos_fp,
                  classify_neg_fp)


if __name__ == '__main__':
    # ner_data_fn = 'data_parallel.txt'
    # ner_data_neg_fn = 'data_parallel_neg.txt'
    # new_ner_data_fn = 'data_parallel_e.txt'
    # # add_pos_events(new_ner_data_fn)
    # # gen_neg_by_other(r_parallel)
    # filter_events()
    gen_relation(r_further, articles_db_path, True)
    relation = r_parallel
