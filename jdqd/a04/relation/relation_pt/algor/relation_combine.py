
# def extract_all_keywords(sentence, rule, keywords):
#   for k in keywords:
#     rst = rule(sentence, k)
#     if rst:
#       return rst
#   return {}


def extract_all_rules(sentence, rules, keyword_rules):
    """
  使用某种关系下
  :param sentence:
  :param rules:
  :param keyword_rules:
  :return:
  """
    for r in rules:
        rule_name = r.__name__
        keywords = keyword_rules[rule_name]
        for kw in keywords:
            extract_rst = r(sentence, kw)
            if extract_rst:
                return extract_rst
    return {}


def extract_all_relations(sentence, relations):
    rsts = []
    for r in relations:
        rules = r.rules
        keyword_rules = r.keyword_rules
        rst = extract_all_rules(sentence, rules, keyword_rules)
        if rst:
            rsts.append(rst)
    return rsts


def split(sentence, keyword, rules, keyword_rules, rules_keyword_pos,
          keyword_pos):
    for r in rules:
        rule_name = r.__name__
        keywords = keyword_rules[rule_name]
        if keyword in keywords:
            rule_keyword_pos = rules_keyword_pos[rule_name]
            if rule_keyword_pos is not None and rule_keyword_pos != keyword_pos:
                continue
            split_rst = r(sentence, keyword)
            if split_rst:
                return split_rst, rule_name
    return {}, None
