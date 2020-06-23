from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern


def gen_keyword_rules():
  is_words = ['属于', '是', '系']
  category_words = ['一种', '一类', '一个']
  keywords_is = [[''.join(wp)] for wp in
                 list(product(is_words, category_words))]
  contain_words = ['包括', '包含', '有']

  keywords3 = list(product(contain_words, ['等']))

  keywords201 = list(product(contain_words, ['在内的']))
  keyword_rules = {'rule101': keywords_is,
                   'rule102': keywords_is,
                   'rule201': keywords201,
                   'rule3': keywords3
                   }
  return keyword_rules


keyword_rules = gen_keyword_rules()


def rule101(sentence, keyword):
  # 匹配模式: ..., 是一种...
  return pattern.rule_scks(sentence, keyword)


def rule102(sentence, keyword):
  # 匹配模式: ...是一种...
  return pattern.rule_sks(sentence, keyword)


def rule201(sentence, keyword):
  # 匹配模式: 包括...在内的...
  return pattern.rule_kscks(sentence, keyword, comma=False)


def rule3(sentence, keyword):
  # 匹配模式: ..., 包括...等
  return pattern.rule_scksk(sentence, keyword, reverse=True)


rules = [rule101, rule102, rule201, rule3]
