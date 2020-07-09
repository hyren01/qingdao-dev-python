# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:57:41 2020

@author: 12894
"""
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Input, Lambda, Dense
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

from jdqd.common.relation_com.model_utils import generate_trained_model_path, get_bert_tokenizer
from jdqd.a04.relation.relation_extract.algor.train.utils.relation_extract_data_util import data_generator, get_data
from feedwork.utils import logger
import jdqd.a04.relation.relation_extract.config.relation_extract_config as CONFIG

import tensorflow as tf
g0 = tf.Graph()
ss0 = tf.Session(graph=g0)

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = generate_trained_model_path(CONFIG.trained_model_dir, CONFIG.trained_model_name)

# 加载训练集和验证集
train_line, test_line = get_data(CONFIG.total_data_path, CONFIG.relation)

# 加载bert分字器
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)

def build_bert():
    """
    构建模型主体
    :param nclass:（int） 类别数
    
    return 模型对象
    """
    #加载bert模型主体
    with ss0.as_default():
        with ss0.graph.as_default():
            bert_model = load_trained_model_from_checkpoint(CONFIG.config_path, CONFIG.checkpoint_path, seq_len=None)  #加载预训练模型
            for l in bert_model.layers:
                l.trainable = True

            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
            p = Dense(3, activation='softmax')(x)
            model = Model([x1_in, x2_in], p)
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(1e-5),    #用足够小的学习率
                          metrics=['accuracy'])
    return model


model = build_bert()

class Evaluate(Callback):
    """
    继承Callback类，改下内部方法，使得当随着训练步数增加时，选择并保存最优模型
    """
    def __init__(self):
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        """
        选择所有训练次数中，f1最大时的模型
        :param epoch: 训练次数
        
        return None
        """
        p, r, f1 = self.evaluate()
        if f1 > self.best:
            self.best = f1
            model.save(TRAINED_MODEL_PATH, include_optimizer=True)
        logger.info(f'epoch: {epoch}, p: {p}, r: {r}, f1: {f1}, best: {self.best}\n')

    def evaluate(self):
        """
        构造模型输出的f1、p、r指标
        
        return f1(float), p(float), r(float)
        """
        true_lists = []
        probs = []
        for i in range(len(test_line)):
            test1 = test_line[i][0]
            test2 = test_line[i][1]
            true_lists.append(np.argmax(test_line[i][2]))
            t1, t1_ = TOKENIZER.encode(first=test1, second=test2)
            T1, T1_ = np.array([t1]), np.array([t1_])
            with ss0.as_default():
                with ss0.graph.as_default():
                    _prob = model.predict([T1, T1_])
            prob = np.argmax(_prob)
            probs.append(prob)

        p = precision_score(true_lists,probs,average='macro') 
        r = recall_score(true_lists,probs,average='macro') 
        f1 = f1_score(true_lists,probs,average='macro') 
        return p, r, f1

def model_test(test_line):
    """
    对测试数据进行预测
    :param input_test:测试集数据
    
    return None
    """
    with open (CONFIG.pred_path, 'w', encoding = 'utf-8') as f:
        for i in range(len(test_line)):
            test1 = test_line[i][0]
            test2 = test_line[i][1]
            t1, t1_ = TOKENIZER.encode(first=test1, second=test2)
            T1, T1_ = np.array([t1]), np.array([t1_])
            with ss0.as_default():
                with ss0.graph.as_default():
                    _prob = model.predict([T1, T1_])
            prob = np.argmax(_prob)
            f.write(test1 + '\t' + test2 + '\t' + str(np.argmax(test_line[i][2])) + '\t' + str(prob) + '\n')
            

def model_train():
    """
    进行模型训练
    :return: None
    """
    train_D = data_generator(TOKENIZER, CONFIG.maxlen, train_line, CONFIG.batch_size)
    valid_D = data_generator(TOKENIZER, CONFIG.maxlen, test_line, CONFIG.batch_size)
    
    # 构造callback模块的评估类
    evaluator = Evaluate()
    
    # 模型训练
    with ss0.as_default():
        with ss0.graph.as_default():
            model.fit_generator(
                                train_D.__iter__(),
                                steps_per_epoch=len(train_D),
                                epochs= CONFIG.epoch,
                                validation_data=valid_D.__iter__(),
                                validation_steps=len(valid_D),
                                verbose=1,
                                shuffle=True,
                                callbacks=[evaluator]
            )
    model_test(test_line)
    p, r, f1 = evaluator.evaluate()
    logger.info(f"f1:{f1}, precision:{p}, recall:{r}")


if __name__ == '__main__':

    model_train()

