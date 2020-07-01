# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from jdqd.a03.event_pred.algor.common import preprocess as pp
from feedwork.utils.FileHelper import cat_path


def gen_samples(values_pca, events_p_oh, input_len, output_len, dates, pred_start_date, pred_end_date):
    """
      根据预测日期生成对应的输入与输出样本
    Args:
      values_pca: pca 降维操作后的数据
      events_p_oh: 按数据表补0的事件列表, one-hot形式
      input_len: encoder 输入序列长度
      output_len: decoder 输出序列长度
      dates: 数据表对应日期列表
      pred_start_date: 开始预测日期
      pred_end_date: 预测截止日期

    Returns:
      输出序列在开始预测日期与预测截止日期之间的样本, 包含输入输出序列, 以及 decoder 训练阶段
      的 inference 输入
    """
    inputs_train, outputs_train = pp.gen_samples_by_pred_date(values_pca, events_p_oh, input_len, output_len,
                                                              dates, pred_start_date, pred_end_date)
    # 训练阶段 inference 输入, 为样本输出序列标签延后一个时间单位, 开头以 0 填充
    # 在事件表与数据表进行join合并的时候，数据表中的某个特征可能在事件表中没有记录该特征的事件类型，
    # 意味着该特征指没有事件发生，所以填充0
    outputs_train_inf = np.insert(outputs_train, 0, 0, axis=-2)[:, :-1, :]
    return inputs_train, outputs_train, outputs_train_inf


def build_models(latent_dim, n_input, n_output):
    """
    构建模型
    Args:
      latent_dim:
      n_input: encoder 输入序列长度
      n_output: decoder 输出长度

    Returns:
      构建好的 encoder-decoder 模型, 以及单独的 encoder 及 decoder 模型
    """
    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量
    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                     initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model


def train(batch_size, epochs, latent_dim, array_x, array_y, array_yin, model_sub_dir):
    """
    训练模型并保存模件
    Args:
      array_x: encoder 的输入序列
      array_y: deocder 的输出序列
      array_yin: decoder 的 inference 序列
      model_sub_dir: 存放此次训练所生成的模型的目录
    """
    # x的shape是[样本数, encoder输入长度(即滞后期), 特征数]
    # y的shape是[样本数, decoder输出长度(即预测天数), 输出事件类别个数]
    n_input = array_x.shape[2]
    n_output = array_y.shape[2]
    model, encoder, decoder = build_models(latent_dim, n_input, n_output)
    model.fit([array_x, array_yin], array_y, batch_size=batch_size, epochs=epochs, verbose=2)
    encoder_path = cat_path(model_sub_dir, 'encoder.h5')
    decoder_path = cat_path(model_sub_dir, 'decoder.h5')
    encoder.save(encoder_path)
    decoder.save(decoder_path)
