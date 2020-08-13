from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern

repeat_words = ['一边是', '一边', '有的是', '有的', '有时候', '有时', '一会儿', '一会', '一面', '也']
repeat_pairs = list(product(repeat_words, repeat_words))
repeat_pairs.append(['一方面', '另一方面'])
parallel_pairs = [['既是', '又是'], ['既是', '也是'], ['既', '又'], ['既', '也']]
parallel_pairs.extend(repeat_pairs)
single_conjs = [['与此相同的是'], ['与之类似的是'], ['与之类似'], ['类似的是'], ['类似地'],
                ['与此相同'], ['与此同时'], ['同时'], ['同样的是'], ['同样'], ['另外'], ['另一方面']]

keyword_rules = {'rule101': parallel_pairs,
                 'rule102': parallel_pairs,
                 'rule103': parallel_pairs,
                 'rule104': parallel_pairs,
                 'rule201': single_conjs,
                 'rule202': single_conjs,
                 'rule203': single_conjs,
                 'rule204': single_conjs
                 }


def rule101(sentence, sub_sentences):
    # 匹配模式: ...一边..., 一边...
    return pattern.rule_skscks(sentence, sub_sentences, parallel_pairs)


def rule102(sentence, sub_sentences):
    # 匹配模式: ...一边...一边...
    return pattern.rule_skscks(sentence, sub_sentences, parallel_pairs,
                               comma=False)


def rule103(sentence, sub_sentences):
    # 匹配模式: 一边...一边...
    match_mode = 'short'
    return pattern.rule_kscks(sentence, sub_sentences, parallel_pairs)


def rule104(sentence, sub_sentences):
    # 匹配模式: 一边...一边...
    match_mode = 'short'
    return pattern.rule_kscks(sentence, sub_sentences, parallel_pairs,
                              comma=False)


def rule201(sentence, sub_sentences):
    # 匹配模式: ..., 同时, ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs)


def rule202(sentence, sub_sentences):
    # 匹配模式: ..., 同时...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs,
                              comma2=False)


def rule203(sentence, sub_sentences):
    # 匹配模式: ...同时, ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs,
                              comma1=False)


def rule204(sentence, sub_sentences):
    # 匹配模式: ...同时 ...
    return pattern.rule_sckcs(sentence, sub_sentences, parallel_pairs,
                              comma1=False, comma2=False)


rules = [rule101, rule102, rule103, rule104, rule201, rule202, rule203, rule204]

rules_keyword_pos = {'rule101': None,
                     'rule102': None,
                     'rule103': None,
                     'rule104': None,
                     'rule201': None,
                     'rule202': None,
                     'rule203': None,
                     'rule204': None
                     }
rw_type = 'parallel'
