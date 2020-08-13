import re

# period_re = '[.。!！？?]'
comma_re = '[,，:：]'


class RelationPattern:
    """
    关系匹配模式类, 用于根据正则表达式及其对应的句子成分结构进行匹配,
    并抽取出句子中不同的关系成分
    """
    def __init__(self, pattern_reg, parts_order, reverse=False):
        """
        初始化函数
        :param pattern_reg: 正则表达式
        :param parts_order: 句子各个成分对应的正则表达式匹配结果的下标列表, 有五个元素,
        分别为关键词下标列表, 左句下标列表, 右句下标列表, 无关成分下标列表, 标点(句号)下标
        列表.
        :param reverse: parts_order 中左句与右句的匹配结果下标是否交换顺序.
        """
        self.pattern_reg = pattern_reg
        self.parts_order = parts_order
        if reverse:
            parts_order[1], parts_order[2] = parts_order[2], parts_order[1]
        self.pattern = re.compile(self.pattern_reg)

    def match_pattern(self, sentence):
        """
        根据生成的正则表达式字符串对句子进行匹配, 并将匹配的结果按照parts_order中定义的各个
        成分的下标组合成关联词, 左右句, 无关成分以及标点, 并以此计算出关联词在原句中的位置
        :param sentence: 输入句子
        :return: 关联词, 左右句, 无关成分, 标点以及关联词位置的字典
        """
        match_result = self.pattern.findall(sentence)
        parts = ['tag', 'left', 'right', 'irrelevant', 'commas']
        if match_result:
            matched = ''.join(match_result[0])
            match_result_index = sentence.index(matched)
            match_result = match_result[0]
            parts_len = {}
            result = {}
            for p, o in zip(parts, self.parts_order):
                hyphen = '-' if p == 'tag' else ''
                part_content = [match_result[i] for i in o]
                parts_len.update([[i, len(match_result[i])] for i in o])
                part_content = hyphen.join(part_content)
                result[p] = part_content
            # 计算关联词位置
            accu_len = match_result_index
            tag_indexes = {}
            prev_i = 0
            for k, i in enumerate(self.parts_order[0]):
                for j in range(prev_i, i):
                    accu_len += parts_len[j]
                tag_start = accu_len
                accu_len += parts_len[i]
                tag_end = accu_len
                tag_indexes[result['tag'].split('-')[k]] = [tag_start, tag_end]
                prev_i = i + 1
            result['tag_indexes'] = tag_indexes
            return result
        return {}


class Rule(object):
    """
    匹配规则的类. 一个匹配规则对应一个有特定句式, 包含特定关联词的正则表达式. 通过此正则表达式
    对句子进行匹配并生成匹配结果.
    """
    def __init__(self, keyword, pattern_reg, parts_order, reverse):
        """
        初始化函数
        :param keyword: 关联词(或关联词对)
        :param pattern_reg: 正则表达式, 关联词部分使用占位符, 待使用关联词替代
        :param parts_order: 句子各个成分对应的正则表达式匹配结果的下标列表, 有五个元素,
        分别为关键词下标列表, 左句下标列表, 右句下标列表, 无关成分下标列表, 标点(句号)下标
        列表.
        :param reverse: parts_order 中左句与右句的匹配结果下标是否交换顺序.
        """
        self.keyword = keyword
        self.pattern_reg = pattern_reg
        if len(parts_order) == 3:
            parts_order.extend([[], []])
        if len(parts_order) == 4:
            parts_order.append([])
        self.parts_order = parts_order
        if reverse:
            parts_order[1], parts_order[2] = parts_order[2], parts_order[1]

    def extract(self, sentence):
        for w in self.keyword:
            if w not in sentence:
                return {}

        # keywords both exist here
        pattern_reg = self.pattern_reg.format(*self.keyword)

        pattern = RelationPattern(pattern_reg, self.parts_order)
        result = pattern.match_pattern(sentence)
        return result


def rule_scskcscskcs(sentence, keywords, reverse=False, comma2=False, comma4=False):
    # ..., ...因为, ..., ...所以, ...
    parts_order = [[2, 6], [1, 4], [5, 8], [0], [3, 7]]
    pattern_reg = r'(.*cm)(.*)({})(cm2)(.*cm)(.*)({})(cm4)(.*)'
    comma2 = comma_re if comma2 else ''
    comma4 = comma_re if comma4 else ''
    pattern_reg = pattern_reg.replace('cm2', comma2).replace('cm4', comma4)
    pattern_reg = pattern_reg.replace('cm', comma_re)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_skcscskcs(sentence, keywords, reverse=False, comma1=False, comma3=False):
    # ...因为, ..., ...所以, ...
    parts_order = [[1, 5], [0, 3], [4, 7], [], [2, 6]]
    pattern_reg = r'(.*)({})(cm1)(.*cm2)(.*)({})(cm3)(.*)'
    cm1 = comma_re if comma1 else ''
    cm3 = comma_re if comma3 else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', comma_re).replace('cm3', cm3)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_scsksks(sentence, keywords, reverse=False):
    # ..., ...因为...所以...
    parts_order = [[2, 4], [1, 3], [5], [0]]
    pattern_reg = r'(.*cm)(.*)({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_skscks(sentence, keywords, reverse=False, comma=True):
    # ...因为..., 所以...
    parts_order = [[1, 3], [0, 2], [4]]
    if comma:
        pattern_reg = '(.*)({})(.*[,，])({})(.*)'
    else:
        pattern_reg = '(.*)({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_kscks(sentence, keywords, reverse=False, comma=True):
    # 因为..., 所以...
    parts_order = [[0, 2], [1], [3]]
    if comma:
        pattern_reg = r'({})(.*[,，])({})(.*)'
    else:
        pattern_reg = r'({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_sks(sentence, keywords, reverse=False):
    # ...所以...
    parts_order = [[1], [0], [2]]
    pattern_reg = r'(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_kscs(sentence, keywords, reverse=False):
    # 因为。。。，。。。
    parts_order = [[0], [1], [2]]
    pattern_reg = r'^({})(.*?[,，])(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_skcscs(sentence, keywords, reverse=False, comma=False):
    # ...因为, ..., ...
    parts_order = [[1], [0, 3], [4], [], [2]]
    pattern_reg = r'(.*)({})(cm1)(.*?cm2)(.*)'
    cm1 = comma_re if comma else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', comma_re)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_scsks(sentence, keywords, reverse=False, comma=True):
    # 。。。，。。。因为。。。
    parts_order = [[2], [0], [1, 3]]
    pattern_reg = r'(.*cm)(.*?)({})(.*)'
    cm = comma_re if comma else ''
    pattern_reg = pattern_reg.replace('cm', cm)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_sckcs(sentence, keywords, reverse=False, comma1=True, comma2=True):
    # 。。。，因为，。。。
    parts_order = [[1], [0], [3], [], [2]]
    pattern_reg = r'(.*cm1)({})(cm2)(.*)'
    cm1 = comma_re if comma1 else ''
    cm2 = comma_re if comma2 else ''
    pattern_reg = pattern_reg.replace('cm1', cm1).replace('cm2', cm2)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_scskscs(sentence, keywords, reverse=False):
    parts_order = [[2], [1, 3], [4], [0]]
    pattern_reg = r'(.*cm1)(.*)({})(.*?cm2)(.*)'
    pattern_reg = pattern_reg.replace('cm1', comma_re).replace('cm2', comma_re)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_sksk(sentence, keywords, reverse=False):
    # 。。。由。。。导致
    parts_order = [[1, 3], [0], [2]]
    pattern_reg = r'(.*)({})(.*)({})'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_kscsks(sentence, keywords, reverse=False):
    parts_order = [[0, 3], [1], [2, 4]]
    pattern_reg = r'({})(.*?[,，])(.*?)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_scksk(sentence, keywords, reverse=False):
    parts_order = [[1, 3], [2, 4], [0]]
    pattern_reg = r'(.*[,，])({})(.*)({})(.*)'
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


def rule_sksks(sentence, keywords, reverse=False):
    parts_order = [[2, 4], [1, 3], [5], [0]]
    pattern_reg = r'(.*cm)(.*)({})(.*)({})(.*)'.replace('cm', comma_re)
    rule = Rule(keywords, pattern_reg, parts_order, reverse)
    return rule.extract(sentence)


if __name__ == '__main__':
    sentence = '如果女方表示婉拒亲吻或亲密举动时, 男人将做什么应对呢？'
    sentence2 = '研究负责人雷格南特博士说：“虽然‘伤心综合征’不被人们所认识，但我们发现，只要病人在头48小时内充分得到心理及生理的救助，病人的恢复就会非常好，但如果失去这一最佳治疗时间，就会过早夺取病人的生命。'
    keywords = ['如果', '将']
    keywords2 = ['如果', '就']

    rst = rule_kscsks(sentence, [keywords])
    print(rst)
    print([sentence[v[0]: v[1]] for v in rst['tag_indexes'].values()])
