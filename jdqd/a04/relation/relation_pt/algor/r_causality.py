from itertools import product
from jdqd.a04.relation.relation_pt.algor import pattern, relation_util

# TODO(zhxin): 处理关系字组成其他常用词的情况
neg_words = ['不', '不曾', '不会', '并不会', '不可', '不可以', '不可能', '不能', '不能够', '避免', '没有', '并没有']
# TODO(zhxin): 处理关系词前出现以下词语的情况: '可能', '很可能', '成为', '变成', '变为'
# TODO(zhxin): 根据连词(如'但是')找出完整的句子, 以适应精确模式与完整模式的调整


adverbs = ['才', '才会', '曾']


# @todo(zhxin): 自从..., ...就...
# @todo(zhxin): 由于..., 加上..., ...就...
# @todo(zhxin): 老年人的情况就没有那么糟糕了，因为老年人并不是在饥荒期间出生的（饥荒始于冷战时期，俄罗斯政府的补贴在20世纪90年代初结束）
# @todo(zhxin): 屈从于必要和中国的压力，北韩经济正日益成为私营企业蓬勃发展的经济体（因为私营企业蓬勃发展效率更高，提供更好的商品和服务），而国营经营很少改善


def gen_keywords():
    cause_for = ['为了', '出于']
    cause_conjs = ['由于', '因为', '缘于'] + cause_for
    effect_conjs = ['从而', '所以', '为此', '因此', '因而', '故而', '从而', '以致于', '以致', '于是', '那么', '才会']
    effect_verbs = ['导致', '引发', '引起', '致使', '使得']
    effect_verbs = relation_util.add_char(effect_verbs, ['了'])
    # 因为, 所以
    keyword_ywsy = list(product(cause_conjs, effect_conjs))
    keyword_ywsy2 = list(product(cause_conjs, ['而']))
    # 因为
    keyword_yw = [[w] for w in cause_conjs]

    keyword_wl = [[w] for w in cause_for]
    keyword_sy = [[w] for w in effect_conjs]

    # 导致
    keyword_dz = [[w] for w in effect_verbs]
    # 之所以
    keyword_zsy = list(product(['之所以'], ['是' + c for c in cause_conjs]))

    # 原因是
    keyword_yys = ['的原因是', '原因是', '理由是', '借口是']  # 原因是
    keyword_yys.extend(['这是' + c for c in cause_conjs])
    keyword_yys.extend(['是' + c for c in cause_conjs])
    keyword_yys = [[w] for w in keyword_yys]

    # 是由...所...
    keywords_sys = [''.join(p) for p in product(['', '所'], effect_verbs)]
    keywords_sys = list(product(['是由'], keywords_sys))

    # 是...的原因
    keywords_syy = [''.join(p) for p in product(['才是', '是'], effect_verbs)]
    keywords_syy = list(product(keywords_syy, ['的真正原因', '的真实原因', '的原因', '的理由', '的借口']))

    keyword_rules = {'rule10': keyword_ywsy,
                     'rule11': keyword_ywsy,
                     'rule12': keyword_ywsy,
                     'rule13': keyword_ywsy,
                     'rule14': keyword_ywsy,
                     'rule20': keyword_zsy,
                     'rule21': keyword_zsy,
                     'rule30': keyword_ywsy,
                     'rule31': keyword_ywsy2,
                     'rule400': keyword_yw,
                     'rule401': keyword_sy,
                     'rule402': keyword_sy,
                     'rule40': keyword_wl,
                     # 'rule41': keyword_yw,
                     'rule42': keyword_yys,
                     'rule50': keyword_yw,
                     'rule51': keyword_yw,
                     'rule52': keyword_yw,
                     'rule53': keyword_yw,
                     'rule60': keywords_sys,
                     'rule70': keyword_dz,
                     'rule80': keywords_syy, }

    return relation_util.tuple_to_list(keyword_rules)


keyword_rules = gen_keywords()


def rule10(sentence, keyword):
    # 匹配模式: ..., ...因为..., 所以, ...
    return pattern.rule_scskcscskcs(sentence, keyword, comma4=True)


def rule11(sentence, keyword):
    # 匹配模式: ..., ...因为..., ...所以...
    return pattern.rule_scskcscskcs(sentence, keyword)


def rule12(sentence, keyword):
    # 匹配模式: ...因为..., ...所以, ...
    return pattern.rule_skcscskcs(sentence, keyword, comma3=True)


def rule13(sentence, keyword):
    # 匹配模式: ...因为..., ...所以...
    return pattern.rule_skcscskcs(sentence, keyword)


def rule14(sentence, keyword):
    # 匹配模式: ..., ...因为...所以...
    return pattern.rule_scsksks(sentence, keyword)


def rule20(sentence, keyword):
    # 匹配模式: ..., ...之所以..., ...是因为...
    return pattern.rule_scskcscskcs(sentence, keyword, reverse=True)


def rule21(sentence, keyword):
    # 匹配模式: ...之所以..., ...是因为...
    return pattern.rule_skcscskcs(sentence, keyword, reverse=True)


def rule30(sentence, keyword):
    # 匹配模式: ...因为...所以...
    return pattern.rule_sksks(sentence, keyword)


def rule31(sentence, keyword):
    # 匹配模式: ...因为...而...
    return pattern.rule_sksks(sentence, keyword)


def rule400(sentence, keyword):
    # 匹配模式: ..., 因为, ...
    return pattern.rule_sckcs(sentence, keyword, reverse=True)


def rule401(sentence, keyword):
    # 匹配模式: ..., 所以, ...
    return pattern.rule_sckcs(sentence, keyword)


def rule402(sentence, keyword):
    # 匹配模式: ...所以...
    return pattern.rule_sckcs(sentence, keyword, comma1=False, comma2=False)


def rule40(sentence, keyword):
    # 匹配模式: ..., ...为了..., ...
    return pattern.rule_scskscs(sentence, keyword)


# def rule41(sentence, keyword):
#     # 匹配模式: ...因为...
#     return pattern.rule_sks(sentence, keyword, reverse=True)


def rule42(sentence, keyword):
    # 匹配模式: ...原因是...
    return pattern.rule_sks(sentence, keyword, reverse=True)


# todo(zhxin): ...结果是...
# todo(zhxin): 以达到...的目的...


def rule50(sentence, keyword):
    # 匹配模式: ...因为..., ...
    # pos: 0
    return pattern.rule_skcscs(sentence, keyword)


def rule51(sentence, keyword):
    # 匹配模式: ..., ...因为..., ...
    # pos: -2
    return pattern.rule_scskscs(sentence, keyword)

def rule52(sentence, keyword):
    # 匹配模式: ..., 因为...
    # pos: -1
    return pattern.rule_sckcs(sentence, keyword, comma2=False, reverse=True)


def rule53(sentence, keyword):
    # 匹配模式: ...因为...
    # pos: None
    return pattern.rule_sckcs(sentence, keyword, comma1=False, comma2=False, reverse=True)

def rule60(sentence, keyword):
    # 模式: ...是由...所引起
    return pattern.rule_sksk(sentence, keyword, reverse=True)


def rule70(sentence, keyword):
    # 匹配模式:...导致...
    return pattern.rule_sks(sentence, keyword)


def rule80(sentence, keyword):
    # 匹配模式: ...是...的原因
    return pattern.rule_sksk(sentence, keyword)


rules = [rule10, rule11, rule12, rule13, rule14, rule20, rule21, rule30, rule31, rule400, rule401,
         rule402, rule40, rule42, rule50, rule51, rule52, rule53, rule60, rule70, rule80]

rules_keyword_pos = {'rule10': None,
                     'rule11': None,
                     'rule12': None,
                     'rule13': None,
                     'rule14': None,
                     'rule20': None,
                     'rule21': None,
                     'rule30': None,
                     'rule31': -1,
                     'rule400': None,
                     'rule401': None,
                     'rule402': None,
                     'rule40': None,
                     # 'rule41': -1,
                     'rule42': None,
                     'rule50': 0,
                     'rule51': -2,
                     'rule52': -1,
                     'rule53': None,
                     'rule60': None,
                     'rule70': None,
                     'rule80': None,
                     }

rw_type = 'causality'

# TODO '王生分析，朝鲜向来不屈服于外界的制裁压力，而且以往的制裁措施对朝鲜本身的影响也十分有限，“朝鲜还是会根据自己的步子进行卫星发射，现在需要判断的是金正恩何时访华，有可能会在明年中国"两会"之后，这不仅是因为到时中国领导人完成了政府和党两个层面的新老交接，也是因为届时美国奥巴马政府和韩国新总统的对朝政策都会明朗起来，目前金正恩主要还是在考虑如何巩固领导地位。”'
# TODO “朝鲜所有行动都是基于国家生存而采取的自卫措施，并不是为了威胁别人。”
# TODO “美海军之所以采取如此做法，一方面是如果全部是军方人员，将引起侦察对象国的强烈反应，而大量使用民事人员则具有很强的隐蔽性和欺骗性”
# TODO(zhxin): ['对于安倍的这种做法，日本国内一直有舆论认为，名义上日本是为了应对朝鲜不断的导弹威胁，但实际上日本政府有两个目的：一是渲染紧张局势，借机加强军备，构建新的日美导弹防御体系，推进日美在东亚地区的军事一体化；另一个更隐秘的目的是为修宪进行舆论上的准备。', {'tag': '是为了', 'cause': '应对朝鲜不断的导弹威胁，但实际上日本政府有两个目的：一是渲染紧张局势，借机加强军备，构建新的日美导弹防御体系，推进日美在东亚地区的军事一体化；另一个更隐秘的目的是为修宪进行舆论上的准备。', 'effect': '对于安倍的这种做法，日本国内一直有舆论认为，名义上日本'}]


if __name__ == '__main__':
    sentence = '我是你爸爸,所以,你是我儿子.'
    rule = rule401
    kw = keyword_rules.get(rule.__name__)
    kw_pos = rules_keyword_pos.get(rule.__name__)
    print(kw)
    for k in kw:
        rst = rule(sentence, k)
        if rst:
            print(rst)
    k = ['因为', '所以']

