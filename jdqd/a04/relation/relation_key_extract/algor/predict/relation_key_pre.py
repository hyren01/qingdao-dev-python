# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:41:25 2020

@author: 12894
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
import tensorflow as tf

from jdqd.common.relation_com.model_utils import get_bert_tokenizer
from jdqd.a04.relation_key_extract.algor.train.utils.relation_key_extract_data_util import get_label, Vector2Id, Id2Label
import jdqd.a04.relation_key_extract.config.relation_extract_config as CONFIG

# 使用bert字典，构建tokenizer类
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)

#加载标签字典
label, _label = get_label(CONFIG.label_path)



# 预处理输入数据
def PreProcessInputData(text):
    """
    输入句子进行Token
    :param test: 输入句子字符串
    
    return 句子Token
    """
    word_labels = []
    seq_types = []
    for sequence in text:
        code = TOKENIZER.encode(first=sequence, max_len=CONFIG.maxlen)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


# 预处理结果数据
def PreProcessOutputData(text):
    """
    输出句子生成标注序列
    :param test: 输出句子字符串
    
    return 句子标注序列
    """
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=CONFIG.maxlen, padding="post", truncating="post")
    result_tags = np.expand_dims(pad_tags, 2)
    return result_tags


def load_model():
    """
    构建模型主体
    return 模型对象
    """
    bert = load_trained_model_from_checkpoint(CONFIG.config_path, CONFIG.checkpoint_path, seq_len=CONFIG.maxlen)
    x1 = Input(shape=(None,))
    x2 = Input(shape=(None,))
    bert_out = bert([x1, x2])
    lstm_out = Bidirectional(LSTM(CONFIG.lstmDim,
                                  return_sequences=True,
                                  dropout=0.2,
                                  recurrent_dropout=0.2))(bert_out)
    crf_out = CRF(len(label), sparse_target=True)(lstm_out)
    model = Model([x1, x2], crf_out)
    model.load_weights(CONFIG.relation_key_extract_model_path)
    return model

def extract_items(sentence):
    """
    预测序列标注
    :param sentence: 输入句子字符串
    
    return 序列标注
    """
    sentence = sentence[:CONFIG.maxlen - 1]
    labels, types = PreProcessInputData([sentence])
    tags = model.predict([labels, types])[0]
    result = []
    for i in range(1, len(sentence) + 1):
        result.append(tags[i])
    result = Vector2Id(result)
    tag = Id2Label(result)
    return tag

if __name__ == '__main__':
    model = load_model()
    sentence = '因为罗兴亚人问题和事件本身可以说涉及到人权问题美国会根据美国自己在这方面的想法或者制度来制定相关的政策'   
    tag  = extract_items(sentence)
    
    single_pr, causes_pr, ends_pr = '', '', ''
    for s, t in zip(sentence, tag):
        if t in ('B-S', 'I-S'):
            single_pr += ' ' + s if (t == 'B-S') else s
        if t in ('B-C', 'I-C'):
            causes_pr += ' ' + s if (t == 'B-C') else s
        if t in ('B-E', 'I-E'):
            ends_pr += ' ' + s if (t == 'B-E') else s
    single_pr = set(single_pr.split())
    causes_pr = set(causes_pr.split())
    ends_pr = set(ends_pr.split())
      
    print(f'sentence: {sentence}' + '\n' + f'single: {single_pr}' + '\t' + f'causes_pr {causes_pr}' + '\t' + f'ends_pr: {ends_pr}')