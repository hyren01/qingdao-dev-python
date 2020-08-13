# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:16:49 2020

@author: 12894
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from jdqd.common.relation_com.model_utils import generate_trained_model_path, get_bert_tokenizer
from jdqd.a04.relation.relation_key_extract.algor.train.utils.relation_key_extract_data_util import get_label, get_data, \
    Vector2Id, Id2Label, cal_precision, cal_recall, cal_f1_score, find_all_tag
from feedwork.utils import logger
import jdqd.a04.relation.relation_key_extract.config.relation_key_extract_config as CONFIG
import tensorflow as tf


g0 = tf.Graph()
ss0 = tf.Session(graph=g0)

# 创建训练后的保存路径,按照当前日期创建模型保存文件夹
TRAINED_MODEL_PATH = generate_trained_model_path(CONFIG.trained_model_dir, CONFIG.trained_model_name)
# 加载字典编码
label, _label = get_label(CONFIG.label_path)
# 加载训练集和验证集
input_train, result_train, input_test, result_test = get_data(CONFIG.train_data_path, CONFIG.dev_data_path,
                                                              CONFIG.relation)
# 加载bert分字器
TOKENIZER = get_bert_tokenizer(CONFIG.dict_path)


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


def build_model():
    '''
    #构建模型主体
    return 模型对象
    '''
    with ss0.as_default():
        with ss0.graph.as_default():
            bert = load_trained_model_from_checkpoint(CONFIG.config_path, CONFIG.checkpoint_path, seq_len=CONFIG.maxlen)
            # 构造模型网络
            for layer in bert.layers:
                layer.trainable = True

            x1 = Input(shape=(None,))
            x2 = Input(shape=(None,))
            bert_out = bert([x1, x2])
            lstm_out = Bidirectional(LSTM(CONFIG.lstmDim,
                                          return_sequences=True,
                                          dropout=0.2,
                                          recurrent_dropout=0.2))(bert_out)
            crf_out = CRF(len(label), sparse_target=True)(lstm_out)
            model = Model([x1, x2], crf_out)
            model.summary()
            model.compile(
                optimizer=Adam(1e-4),
                loss=crf_loss,
                metrics=[crf_accuracy]
            )
    return model


model = build_model()


def extract_items(sentence):
    """
    预测序列标注
    :param sentence: 输入句子字符串

    return 序列标注
    """
    sentence = sentence[:CONFIG.maxlen - 1]
    labels, types = PreProcessInputData([sentence])
    with ss0.as_default():
        with ss0.graph.as_default():
            tags = model.predict([labels, types])[0]
    result = []
    for i in range(1, len(sentence) + 1):
        result.append(tags[i])
    result = Vector2Id(result)
    tag = Id2Label(result, _label)
    return tag


class Evaluate(Callback):
    """
    继承Callback类，改下内部方法，使得当随着训练步数增加时，选择并保存最优模型
    """

    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        选择所有训练次数中，f1最大时的模型
        :param epoch: 训练次数

        return None
        """
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            model.save(TRAINED_MODEL_PATH, include_optimizer=True)
        logger.info(f'epoch: {epoch}, f1: {f1}, precision: {precision}, recall: {recall}, best f1: {self.best}\n')

    @staticmethod
    def flat_lists(lists):
        """
        对传入的列表进行拉平
        :param lists: 传入二维列表

        :return: 拉平后的以后列表
        """
        all_elements = []
        for lt in lists:
            all_elements.extend(lt)
        return all_elements

    def evaluate(self):
        """
        构造模型输出的f1、precision、recall指标

        return f1(float), precision(float), recall(float)
        """
        p_list = []
        r_list = []
        f1_list = []
        for i in range(len(input_test)):
            input_line = input_test[i]
            result_line = result_test[i]
            tag = extract_items(input_line)
            p_ = cal_precision(tag, result_line)
            r_ = cal_recall(tag, result_line)
            f1_list.append(cal_f1_score(p_, r_))
            p_list.append(p_)
            r_list.append(r_)
        f1 = np.mean(f1_list)
        precision = np.mean(p_list)
        recall = np.mean(r_list)
        return f1, precision, recall


def model_test(input_test):
    """
    输出测试结果
    :param input_test:测试集数据
    return 测试结果保存到文件，无返回值
    """
    with open(CONFIG.pred_path, 'w', encoding='utf-8') as f:
        for i in range(len(input_test)):
            input_line = input_test[i]
            result_line = result_test[i]
            tag = extract_items(input_line)
            true_key = find_all_tag(result_line)
            pre_key = find_all_tag(tag)
            true_str = ''
            pre_str = ''
            for key in true_key:
                if true_key[key]:
                    for k in true_key[key]:
                        wd_stt, wd_len = k[0], k[1]
                        kw = input_line[wd_stt, wd_stt + wd_len]
                        str_add = kw + str(k) + '\t'
                        true_str += str_add
            for key in pre_key:
                if pre_key[key]:
                    for k in pre_key[key]:
                        wd_stt, wd_len = k[0], k[1]
                        kw = input_line[wd_stt, wd_stt + wd_len]
                        str_add = kw + str(k) + '\t'
                        pre_str += str_add
            f.write('sentence: ' + input_line + '\n' + 'tru: ' + str(true_str) + '\n' + 'pre: ' + str(pre_str) + '\n')


def model_train():
    """
    进行模型训练
    :return: None
    """
    # 训练集输入输出
    input_train_labels, input_train_types = PreProcessInputData(input_train)
    result_train_pro = PreProcessOutputData(result_train)
    # 测试集输入输出
    input_test_labels, input_test_types = PreProcessInputData(input_test)
    result_test_pro = PreProcessOutputData(result_test)
    # 构造callback模块的评估类
    evaluator = Evaluate()

    # 模型训练
    with ss0.as_default():
        with ss0.graph.as_default():
            model.fit(x=[input_train_labels, input_train_types],
                      y=result_train_pro,
                      batch_size=CONFIG.batch_size,
                      epochs=CONFIG.epochs,
                      validation_data=[[input_test_labels, input_test_types], result_test_pro],
                      verbose=1,
                      shuffle=True,
                      callbacks=[evaluator]
                      )
    model_test(input_test)
    f1, precision, recall = evaluator.evaluate()
    logger.info(f"f1:{f1}, precision:{precision}, recall:{recall}")


if __name__ == '__main__':
    model_train()





