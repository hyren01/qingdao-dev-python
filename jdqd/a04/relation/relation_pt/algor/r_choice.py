from jdqd.a04.relation.relation_pt.algor import pattern

choice_words1 = ['要是', '与其']
choice_words2 = ['还不如', '倒不如', '不如']

word_pairs = [['不是', '就是'], ['宁可', '也不'], ['宁愿', '也不'],
              ['不是', '就是'], ['不是', '即是'], ['或者', '或者']]


def rule101(sentence, keyword):
    # 匹配模式: ...与其..., ...不如...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule102(sentence, keyword):
    # 匹配模式: ...与其..., 不如...
    return pattern.rule_skscks(sentence, keyword)


def rule103(sentence, keyword):
    # 匹配模式: ...与其...不如...
    return pattern.rule_skscks(sentence, keyword, comma=False)


def rule104(sentence, keyword):
    # 匹配模式: 与其..., ...不如...
    return pattern.rule_kscsks(sentence, keyword)


def rule105(sentence, keyword):
    # 匹配模式: 与其..., 不如...
    return pattern.rule_kscks(sentence, keyword)


def rule301(sentence, keyword):
    # 匹配模式: ..., ...不如...
    return pattern.rule_scsks(sentence, keyword)


def rule302(sentence, keyword):
    # 匹配模式: ..., 不如...
    return pattern.rule_sckcs(sentence, keyword, comma2=False)


rules = [rule101, rule102, rule103, rule104, rule105, rule301, rule302]

keyword_rules = {k.__name__: word_pairs for k in rules}

