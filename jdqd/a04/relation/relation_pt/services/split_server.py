from jdqd.a04.relation.relation_pt.algor import relation_combine
from jdqd.a04.relation.relation_pt.algor import r_causality, r_assumption, \
    r_condition, pattern

relations = [r_causality, r_assumption, r_condition]


def split_by_relation(relation, sentence, keyword, keyword_pos):
    return relation_combine.split(sentence, keyword, relation.rules,
                                  relation.keyword_rules,
                                  relation.rules_keyword_pos, keyword_pos)


def split_by_relations(sentence, keyword, keyword_pos):
    """
    遍历条件假设因果三种关系使用关键词进行分句
    :param sentence: 输入
    :param keyword: 关系关联词
    :param keyword_pos: 关联词在句子中的子句的位置, 用于判断
    :return:
    """
    for r in relations:
        split_rst, __ = split_by_relation(r, sentence, keyword, keyword_pos)
        left = split_rst['left']
        right = split_rst['right']
        if left and right:
            return [[left, right]]
    return None


def split_multi_keywords(sentence, keyword):
    """
    如果关键词为一对(e.g. [因为, 所以]), 则调用此方法. 此方法列出各种关键词对对应的分句规则,
    如果符合其一, 则返回该规则分句结果. 都不满足则返回None
    :param sentence: 输入句子
    :param keyword: 关联词对
    :return:
    """
    # ..., ...因为, ..., ...所以, ...
    rst = pattern.rule_scskcscskcs(sentence, keyword, comma2=True, comma4=True)
    if rst:
        return [[rst['left'], rst['right']]]

    # ..., ...因为, ..., ...所以...
    rst = pattern.rule_scskcscskcs(sentence, keyword, comma2=True)
    if rst:
        return [[rst['left'], rst['right']]]

    # ..., ...因为..., ...所以, ...
    rst = pattern.rule_scskcscskcs(sentence, keyword, comma4=True)
    if rst:
        return [[rst['left'], rst['right']]]

    # ..., ...因为..., ...所以 ...
    rst = pattern.rule_scskcscskcs(sentence, keyword)
    if rst:
        return [[rst['left'], rst['right']]]

    # ...因为, ..., ...所以, ...
    rst = pattern.rule_skcscskcs(sentence, keyword, comma1=True, comma3=True)
    if rst:
        return [[rst['left'], rst['right']]]
    # ...因为 ..., ...所以, ...
    rst = pattern.rule_skcscskcs(sentence, keyword, comma3=True)
    if rst:
        return [[rst['left'], rst['right']]]
    # ...因为, ..., ...所以 ...
    rst = pattern.rule_skcscskcs(sentence, keyword, comma1=True)
    if rst:
        return [[rst['left'], rst['right']]]
    # ...因为 ..., ...所以...
    rst = pattern.rule_skcscskcs(sentence, keyword)
    if rst:
        return [[rst['left'], rst['right']]]

    # ...因为...所以...
    rst = pattern.rule_sksks(sentence, keyword)
    if rst:
        return [[rst['left'], rst['right']]]
    return None


def restore_sentence(min_sentences, delimiters):
    return ''.join([''.join(z) for z in zip(min_sentences, delimiters)])


def split_single_keyword(sentence, keyword, min_sentences, delimiters,
                         keyword_pos):
    min_sentences_num = len(min_sentences)

    if min_sentences_num == 1:
        return [sentence.split(keyword[0])]
    if min_sentences_num == 2:
        return [min_sentences]

    split_rsts = []
    # 以keyword划分
    left = restore_sentence(min_sentences[:keyword_pos],
                            delimiters[:keyword_pos])
    right = restore_sentence(min_sentences[keyword_pos:],
                             delimiters[keyword_pos:])

    if left and right:
        split_rsts.append([left, right])

    for split_index in range(keyword_pos + 1, min_sentences_num):
        left = restore_sentence(min_sentences[:split_index],
                                delimiters[:split_index])
        right = restore_sentence(min_sentences[split_index:],
                                 delimiters[split_index:])
        if left and right:
            split_rsts.append([left, right])

    return split_rsts


def split(sentence, keyword):
    keyword_num = len(keyword)
    min_sentences, delimiters = relation_util.split_sentence(sentence)
    if keyword_num == 2:
        keyword_pos = None
    else:
        keyword_pos = relation_util.get_keyword_pos(min_sentences, keyword[0])
    split_rst = split_by_relations(sentence, keyword, keyword_pos)
    if split_rst:
        return split_rst, 'r'

    if keyword_num > 1:
        split_rst = split_multi_keywords(sentence, keyword)
        if split_rst:
            return split_rst, 'm'
    else:
        split_rst = split_single_keyword(sentence, keyword, min_sentences,
                                         delimiters, keyword_pos)
        if split_rst:
            return split_rst, 's'
    return [], ''
