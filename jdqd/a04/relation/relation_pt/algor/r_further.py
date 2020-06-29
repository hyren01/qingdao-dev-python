from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern


def gen_keywords():
    base_words = ['不仅仅', '不仅', '不光', '不但', '固然']
    also_words = ['而且', '并且', '又', '还', '更是', '但更']
    keywords = list(product(base_words, also_words))
    keywords_also = [[w] for w in also_words]

    keywords_ = ['而' + w for w in base_words]
    keywords_ = keywords_ + base_words
    keywords_ = [[w] for w in keywords_]

    keyword_rules = {'rule101': keywords,
                     'rule102': keywords,
                     'rule103': keywords,
                     'rule104': keywords,
                     'rule201': keywords_also,
                     'rule202': keywords_also,
                     'rule301': keywords_
                     }
    return keyword_rules


keyword_rules = gen_keywords()


def rule101(sentence, keyword):
    # 匹配模式: ...不仅..., ...还...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule102(sentence, keyword):
    # 匹配模式: ...不仅..., 而且...
    return pattern.rule_skscks(sentence, keyword)


def rule103(sentence, keyword):
    # 匹配模式: 不仅..., ...而且...
    return pattern.rule_kscsks(sentence, keyword)


def rule104(sentence, keyword):
    # 匹配模式: 不仅..., 而且...
    return pattern.rule_kscks(sentence, keyword)


def rule201(sentence, keyword):
    # 匹配模式: ..., ...而且...
    return pattern.rule_scsks(sentence, keyword)


def rule202(sentence, keyword):
    # 匹配模式: ..., 而且...
    return pattern.rule_sckcs(sentence, keyword, comma2=False)


def rule301(sentence, keyword):
    # 匹配模式: ..., 而不仅仅...
    return pattern.rule_sckcs(sentence, keyword, reverse=True, comma2=False)


rules = [rule101, rule102, rule103, rule104, rule201, rule202, rule301]
