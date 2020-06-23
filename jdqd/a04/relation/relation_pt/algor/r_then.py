from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern

first_words = ['首先', '先是', '先', '第一步', '第一', '事先']
second_words = ['然后', '之后', '紧接着', '接着', '接下来', '再', '第二步', '第二']
keywords = list(product(first_words, second_words))
# keywords.extend(list(product(['一', '才'], ['就'])))
# TODO(zhxin): 一..., ...就...


def rule101(sentence, keyword):
    # 匹配模式: ...首先..., ...然后...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule102(sentence, keyword):
    # 匹配模式: ...首先..., 然后...
    return pattern.rule_skscks(sentence, keyword)


def rule103(sentence, keyword):
    # 匹配模式: ...首先...然后...
    return pattern.rule_skscks(sentence, keyword, comma=False)


def rule104(sentence, keyword):
    # 匹配模式: 首先..., ...然后...
    return pattern.rule_kscsks(sentence, keyword)


def rule105(sentence, keyword):
    # 匹配模式: 首先..., 然后...
    return pattern.rule_kscks(sentence, keyword)


def rule301(sentence, keyword):
    # 匹配模式: ..., ...然后...
    return pattern.rule_scsks(sentence, keyword)


def rule302(sentence, keyword):
    # 匹配模式: ..., 然后...
    return pattern.rule_sckcs(sentence, keyword, comma2=False)


rules = [rule101, rule102, rule103, rule104, rule105, rule301, rule302]

keyword_rules = {k.__name__: keywords for k in rules}
