#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年06月09
"""
分别使用textrank和mmr方法抽取文本摘要
"""
import operator
import jieba
from jdqd.common.event_emm.data_utils import data_process, get_sentences
from sklearn.feature_extraction.text import CountVectorizer
from textrank4zh import TextRank4Sentence
from feedwork.utils import logger


def mmr_subtract(content: str, n = 3):
    """
    传入文章内容，使用mmr方式抽取文章摘要
    :param content: (str) 文章内容
    :return: summary_sentences(list)摘要句子列表
    """

    def encode_sen(sen: str, corpus: list):
        """
        对传入的句子进行向量化
        :param sen: (str)待向量化的句子
        :param corpus: (list)传入的语料
        :return: 词袋法向量化后的句子向量
        """
        cv = CountVectorizer()
        cv = cv.fit(corpus)
        vec = cv.transform([sen]).toarray()

        return vec[0]

    def cosin_distance(vector1: list, vector2: list):
        """
        计算两个向量之间的夹角余弦值
        :param vector1: (list) 向量1
        :param vector2: (list) 向量2
        :return: 夹角余弦值--相似度(float)
        """
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            norm_a += a ** 2
            norm_b += b ** 2
        if norm_a == 0.0 or norm_b == 0.0:
            return 0
        else:
            return dot_product / ((norm_a * norm_b) ** 0.5)

    def doc_list2str(doc_list: list):
        """
        将分词后的句子列表拼接为字符串
        :param doc_list: (list) 分词后的句子列表
        :return: docu_str(str) 字符串
        """
        docu_str = ""
        for wordlist in doc_list:
            temp = " ".join(wordlist)
            docu_str = f"{docu_str} {temp}"

        return docu_str

    def mmr(doc_list, corpus, n = 3):
        """
        传入分词句子列表使用空格分割的句子列表，计算句子权重，返回摘要句子列表
        :param doc_list:(list)分词后句子二维列表 [ [你，喜欢， 喝， 青岛， 啤酒]，  ]
        :param corpus:(list)空格分割后分词的句子列表[你 喜欢 喝 青岛 啤酒，  ]
        :param n: 抽取摘要的句子为n
        :return:summary_set(list) 摘要句子列表
        """
        # 分词后的文章字符串，词之间以空格间隔。
        docu = doc_list2str(doc_list)  # 你 喜欢 喝 青岛 啤酒。
        # 将文章向量化，词袋向量
        doc_vec = encode_sen(docu, corpus)
        # 句子与文章相似度字典
        qd_score = {}
        # 计算句子与文章的相似度作为句子的权重
        for sentence in doc_list:
            # 分词后的句子字符串，词之间以空格间隔。
            sentence = " ".join(sentence)  # 你 喜欢 喝 青岛 啤酒。
            sen_vec = encode_sen(sentence, corpus)
            qd_score[sentence] = cosin_distance(sen_vec, doc_vec)

        # 抽取摘要的句子为n+1
        n = n-1
        # 设置摘要权重的阈值
        alpha = 0.7
        # 摘要句子列表
        summary_set = []
        while n > 0:
            mmr_score = {}
            # 选择权重值最大的作为文章的第一个摘要
            if not summary_set:
                selected = max(qd_score.items(), key=operator.itemgetter(1))[0]
                summary_set.append(selected)

            # 构造摘要语句语料
            summary_set_str = " ".join(summary_set)

            for sentence in qd_score.keys():
                # 计算句子的mmr值
                if sentence not in summary_set:
                    # 摘要向量
                    sum_vec = encode_sen(summary_set_str, corpus)
                    # 句子向量
                    sentence_vec = encode_sen(sentence, corpus)
                    # 计算句子的mmr分数
                    mmr_score[sentence] = alpha * qd_score[sentence] - (1 - alpha) * cosin_distance(sentence_vec,
                                                                                                    sum_vec)
            if mmr_score.items():
                selected = max(mmr_score.items(), key=operator.itemgetter(1))[0]
                summary_set.append(selected)
                n -= 1
            else:
                n -= 1

        return summary_set

    # 将文章分割成句子
    sentence_list = get_sentences(content)
    # 分词后的句子二维列表
    tokened_sentence_list = [jieba.lcut(i) for i in sentence_list if i]
    # 使用" "将词拼接，文章句子列表
    joined_sentence_list = [" ".join(i) for i in tokened_sentence_list if i]
    # 传入分词后的句子列表以及分词后的文章句子列表
    summary_sentences = mmr(tokened_sentence_list, joined_sentence_list, n)

    return summary_sentences


def text_rank_subtract(content: str, n = 3):
    """
    传入中文字符串，使用text rank方法抽取摘要
    :param content: (str)文章内容
    :param n: 抽取摘要的句子数量n
    :return: summary_sentences(list)摘要句子列表
    """
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=content, lower=True, source='all_filters')
    # 抽取摘要
    summary_sentences = [item.sentence for item in tr4s.get_key_sentences(num=n)]

    return summary_sentences


def get_abstract(content: str, n = 3):
    """
    传入清洗后的文章内容，分别使用text rank 和mmr方法提取文章摘要，返回摘要语句列表
    :param content: (str)清洗后的文本字符串
    :return: summary_sentences(list) 摘要句子列表
    :raise: TypeError
    """
    if not isinstance(content, str):
        logger.error("待抽取摘要的内容格式错误！")
        raise TypeError

    # 摘要句子列表
    summary_sentences = []
    # 使用mmr方式抽取的摘要句子列表
    mmr_summary_sentences = mmr_subtract(content, n)
    # 使用text rank抽取的摘要句子列表
    text_rank_summary_sentences = text_rank_subtract(content, n)

    summary_sentences.extend(mmr_summary_sentences)
    summary_sentences.extend(text_rank_summary_sentences)

    # 将句子中的空格剔除
    summary_sentences = [once.replace(" ", "") for once in summary_sentences if once]
    summary_sentences = list(set(summary_sentences))

    return summary_sentences


if __name__ == '__main__':
    text = """韩联社首尔1月1日电 据朝中社1日报道，朝鲜劳动党委员长金正恩在前一日举行的第七届五中全会最后一天会议上作报告时表示，面对敌对势力的制裁压力，朝鲜必须展开正面突破战，为社会主义建设开辟新的出路。 朝鲜2018年4月宣布中断核试与洲际弹道导弹试射，并采取集中力量发展经济路线，金正恩此言代表朝鲜将时隔1年8个月重返“核武开发和经济发展”并进路线。朝中社当天介绍的全体会议结果中，“正面突破”或“正面突破战”的表述出现23次。 综合金正恩的发言，“正面突破”新路线是指不屈服于国际社会的制裁，通过自力更生发展经济，同时继续研发新型战略武器，加强国防力量建设。 金正恩警告，若美国坚持对朝敌对政策，半岛将永远无法实现无核化。此言实际上宣布在短期内不会进行无核化谈判。金正恩还提出，我们决不容许美国恶意利用朝美对话达成动机不纯的目标，我们将以令人震惊的实际行动获取人民受到的痛苦和发展被遏制的代价。 金正恩还暗示可能重启核导活动，他谴责称，朝方曾采取一系列无核化措施，但美方却以韩美军事演习、引进尖端武器、额外制裁等加以回报，并暗示朝鲜今后重启核武和洲际弹道导弹试射。 然而，金正恩还表示将根据美方态度调整应对方式，为朝美对话留余地，敦促美国改变态度。（完） < img src="/p_img/yna_ACK20200101001200881_1.jpg"/> 据朝中社1月1日报道，国务委员会委员长金正恩前一日主持召开劳动党第七届五中全会。 韩联社/朝中社（图片仅限韩国国内使用，严禁转载复制） < img src="/p_img/yna_ACK20200101001200881_2.jpg"/> 据朝中社1月1日报道，国务委员会委员长金正恩前一日主持召开劳动党第七届五中全会。图为会议现场照。 韩联社/朝中社（图片仅限韩国国内使用，严禁转载复制） yhongjing@yna.co.kr 【版权归韩联社所有，未经授权严禁转载复制】 관련뉴스 金正恩今年元旦或不发表新年贺词 详讯：金正恩称将开发战略武器但为核谈留余地 简讯：金正恩称美国越拖时间将只会走投无路 韩统一部：朝鲜祖平统委员长李善权职务未变 朝媒总结2019：唯有自力更生才是生存之路 详讯：金正恩提出需准备进攻性军事外交应对措施 朝鲜劳动党七届五中全会进入第三天 <저작권자(c) 연합뉴스, 무단 전재-재배포 금지> 2020/01/01 13:53 송고
"""

    text = data_process(text)
    # sentences = get_abstract(text, 2)
    sentences = text_rank_subtract(text, 2)
    # sentences = mmr_subtract(text, 2)

    print(sentences)