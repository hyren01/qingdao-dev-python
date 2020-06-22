# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:06:01 2020

@author: 12894
"""
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Input, Lambda, Dense
from keras.models import Model

from jdqd.common.relation_com.model_utils import get_bert_tokenizer
import jdqd.a04.relation_extract.config.relation_extract_config as CONFIG

# 使用bert字典，构建tokenizer类
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)
 

def load_model():
    '''
    构建模型主体
    return 模型对象
    '''
    #搭建bert模型主体
    bert_model = load_trained_model_from_checkpoint(CONFIG.config_path, CONFIG.checkpoint_path, seq_len=None)  #加载预训练模型
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    p = Dense(3, activation='softmax')(x)
    model = Model([x1_in, x2_in], p)
    model.load_weights(CONFIG.relation_extract_model_path)
    return model

    
def class_pre(test1, test2):
    '''
    输入事件对，得到关系结果
    :param test1:事件1
    :param test2:事件2
    return 关系类别
    '''
    t1, t1_ = TOKENIZER.encode(first=test1, second=test2)
    T1, T1_ = np.array([t1]), np.array([t1_])
    _prob = model.predict([T1, T1_])
    prob = np.argmax(_prob)
    return prob

if __name__ == '__main__':
    model = load_model()
    test1 = '这两名工人超过250毫希沃特'	
    test2 = '进行他们,紧急医疗救治'
    prob = class_pre(test1, test2)
    print(prob)
            

