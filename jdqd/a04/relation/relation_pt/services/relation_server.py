from jdqd.a04.relation.relation_pt.algor import relation_combine


def extract(sentence, relation):
    return relation_combine.extract_all_rules(sentence, relation.rules,
                                              relation.keyword_rules)

