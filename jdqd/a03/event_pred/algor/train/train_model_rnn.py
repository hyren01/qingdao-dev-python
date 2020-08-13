# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from feedwork.utils.FileHelper import cat_path
from jdqd.a03.event_pred.algor.train import model_evalution
from jdqd.a03.event_pred.algor.common import preprocess
import numpy as np
from feedwork.utils import logger
from keras.callbacks import Callback
from jdqd.a03.event_pred.algor.predict import predict_rnn


def __build_model(latent_dim, n_input, n_output):
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
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
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
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model


def train(batch_size, epochs, latent_dim, array_x, array_y, array_yin,
          model_dir, data, events_p_oh, input_len, output_len, dates,
          eval_start_date, eval_end_date, events_set):
    """
    训练模型并保存模件
    Args:
      batch_size: int.每批次训练个数
      epochs: int. 训练次数
      latent_dim: int. 网络数
      array_x: encoder 的输入序列
      array_y: deocder 的输出序列
      array_yin: decoder 的 inference 序列
      model_dir: 存放此次训练所生成的模型的目录
    """
    # x的shape是[样本数, encoder输入长度(即滞后期), 特征数]
    # y的shape是[样本数, decoder输出长度(即预测天数), 输出事件类别个数]
    n_input = array_x.shape[2]
    n_output = array_y.shape[2]
    model, encoder, decoder = __build_model(latent_dim, n_input, n_output)

    callback = Evaluate(model_dir, data, events_p_oh, input_len,
                        output_len, dates, eval_start_date, eval_end_date,
                        events_set, encoder, decoder)

    model.fit([array_x, array_yin], array_y, batch_size=batch_size,
              epochs=epochs, callbacks=[callback], verbose=2)


class Evaluate(Callback):
    """
    继承Callback类，改下内部方法，使得当随着训练步数增加时，选择并保存最优模型
    """

    def __init__(self, model_dir, data, events_p_oh, input_len,
                 output_len, dates, eval_start_date, eval_end_date,
                 events_set, encoder, decoder):
        self.best = -1.
        self.model_dir = model_dir
        self.data = data
        self.events_p_oh = events_p_oh
        self.input_len = input_len
        self.output_len = output_len
        self.dates = dates
        self.eval_start_date = eval_start_date
        self.eval_end_date = eval_end_date
        self.events_set = events_set
        self.encoder = encoder
        self.decoder = decoder

    def on_epoch_end(self, epoch, logs=None):
        """
        选择所有训练次数中，f1最大时的模型
        :param epoch: 训练次数

        return None
        """
        score_summary = self.evaluate()
        if score_summary > self.best:
            logger.info(f'{score_summary} better than old: {self.best}')
            self.best = score_summary
            encoder_path = cat_path(self.model_dir, 'encoder.h5')
            decoder_path = cat_path(self.model_dir, 'decoder.h5')
            self.encoder.save(encoder_path)
            self.decoder.save(decoder_path)

    def evaluate(self):
        inputs_test, outputs_test = \
            preprocess.gen_samples_by_pred_date(self.data,
                                                self.events_p_oh,
                                                self.input_len,
                                                self.output_len,
                                                self.dates,
                                                self.eval_start_date,
                                                self.eval_end_date)

        events_num = preprocess.get_event_num(outputs_test, self.events_set)
        preds = predict_rnn.predict_samples(self.encoder, self.decoder,
                                            inputs_test, self.output_len,
                                            len(self.events_set))
        evals_summary, evals_separate, eval_events = \
            model_evalution.evaluate_sub_model(
                preds,
                outputs_test,
                self.events_set,
                events_num)
        return evals_summary[0]
