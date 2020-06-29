from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern, relation_util


def gen_keywords():
    if_words = ['如若', '若', '如果', '假如', '假若', '假使', '假设', '设使', '倘使', '一旦', '要是', '既然']
    keywords_if = [[w] for w in if_words]
    then_words = ['那么', '则', '就将', '便', '将', '就']
    keyword_pairs = list(product(if_words, then_words))
    keyword_rules = {'rule10': keyword_pairs,
                     'rule11': keyword_pairs,
                     'rule20': keyword_pairs,
                     'rule30': keywords_if,
                     'rule31': keywords_if,
                     'rule40': keywords_if,
                     'rule41': keywords_if}

    return relation_util.tuple_to_list(keyword_rules)


keyword_rules = gen_keywords()


def rule10(sentence, keyword):
    # 匹配模式: ..., ...如果..., ...就...
    return pattern.rule_scskcscskcs(sentence, keyword)


def rule11(sentence, keyword):
    # 匹配模式: ...如果..., ...就...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule20(sentence, keyword):
    # 匹配模式: ...如果...就...
    return pattern.rule_sksks(sentence, keyword)


def rule30(sentence, keyword):
    # 匹配模式: ...如果..., ...
    return pattern.rule_skcscs(sentence, keyword)


def rule31(sentence, keyword):
    # 匹配模式: ..., ...如果..., ...
    # pos: -2
    return pattern.rule_scskscs(sentence, keyword)


def rule40(sentence, keyword):
    # 匹配模式: ..., ...如果...
    # pos: -1
    return pattern.rule_scsks(sentence, keyword, reverse=True)


def rule41(sentence, keyword):
    # 匹配模式: ...如果...
    return pattern.rule_sks(sentence, keyword, reverse=True)


rules = [rule10, rule11, rule20, rule30, rule31, rule40, rule41]

rules_keyword_pos = {'rule10': None,
                     'rule11': None,
                     'rule20': None,
                     'rule30': 0,
                     'rule31': -2,
                     'rule40': -1,
                     'rule41': -1}
