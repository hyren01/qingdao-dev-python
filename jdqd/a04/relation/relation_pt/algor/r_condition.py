from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern, relation_util


def gen_keywords():
    # 只要就
    cond_words1 = ['只要']
    result_words1 = ['就将', '便', '就', '才将', '才']
    keywords_zyj = list(product(cond_words1, result_words1))

    # 只有才
    cond_words2 = ['只有', '除非', '除了']
    result_words2 = ['才能', '才会', '才可以', '才']
    keywords_zyc = list(product(cond_words2, result_words2))

    keyword_rules = {'rule10': keywords_zyj,
                     'rule11': keywords_zyj,
                     'rule12': keywords_zyj,
                     'rule13': keywords_zyj,
                     'rule20': keywords_zyc,
                     'rule21': keywords_zyc,
                     'rule22': keywords_zyc,
                     'rule23': keywords_zyc}

    return relation_util.tuple_to_list(keyword_rules)


keyword_rules = gen_keywords()


def rule10(sentence, keyword):
    # 匹配模式: ..., ...只要..., ...就, ...
    return pattern.rule_scskcscskcs(sentence, keyword, comma4=True)


def rule11(sentence, keyword):
    # 匹配模式: ..., ...只要..., ...就...
    return pattern.rule_scskcscskcs(sentence, keyword)


def rule12(sentence, keyword):
    # 匹配模式: ...只要..., ...就, ...
    return pattern.rule_skcscskcs(sentence, keyword, comma3=True)


def rule13(sentence, keyword):
    # 匹配模式: ...只要..., ...就...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule20(sentence, keyword):
    # 匹配模式: ..., ...只有..., ...才能, ...
    return pattern.rule_scskcscskcs(sentence, keyword, comma4=True)


def rule21(sentence, keyword):
    # 匹配模式: ..., ...只有..., ...才能...
    return pattern.rule_scskcscskcs(sentence, keyword)


def rule22(sentence, keyword):
    # 匹配模式: ...只有..., ...才能, ...
    return pattern.rule_skcscskcs(sentence, keyword, comma3=True)


def rule23(sentence, keyword):
    # 匹配模式: ...只有..., ...才能...
    return pattern.rule_skcscskcs(sentence, keyword)


rules = [rule10, rule11, rule12, rule13, rule20, rule21, rule22, rule23]
rules_keyword_pos = {r.__name__: None for r in rules}
