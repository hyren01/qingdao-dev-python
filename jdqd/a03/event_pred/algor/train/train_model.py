# -*- coding: utf-8 -*-
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from jdqd.a03.event_pred.algor.common import preprocess as pp
from feedwork.utils import logger
from feedwork.utils.FileHelper import cat_path


# @todo(zhxin): 每个参数都要有注释，参数列表里必须注明类型，注释需详细
class SeqModel(object):

    """事件预测 Seq2Seq 模型, 即使用 encoder 将输入序列编码成定长语义向量, 使用 decoder
      将语义向量转化为预测输出.
    """
    def __init__(self, n_in, n_out, latent_dim, batch_size, epochs, pca_n):
        """SeqModel 初始化方法
        Args:
          n_in: encoder 输入序列长度, 即一个用于预测的多天数据表连续序列的天数
          n_out: decoder 输出序列长度, 即一个样本的预测天数
          latent_dim: LSTM 单元中隐藏状态的维度
          batch_size: 模型训练时一个批中样本数.
          epochs: 使用所有样本进行训练的次数. 数字越大所需训练时间越长, 模型在训练集上拟合更好
          pca_n: 数据表数据降维维度
        """
        self.n_in = n_in
        self.n_out = n_out
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.pca_n = pca_n

    def gen_samples(self, values_pca, events_p_oh, input_len, output_len, dates, pred_start_date, pred_end_date):
        """
          根据预测日期生成对应的输入与输出样本
          # TODO 对于日期的选择（开始结束日期范围内）
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

    def build_models(self, n_input, n_output):
        """
        构建模型
        Args:
          n_input: encoder 输入序列长度
          n_output: decoder 输出长度

        Returns:
          构建好的 encoder-decoder 模型, 以及单独的 encoder 及 decoder 模型
        """
        # 训练模型中的encoder
        encoder_inputs = Input(shape=(None, n_input))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]  # 仅保留编码状态向量
        # 训练模型中的decoder
        decoder_inputs = Input(shape=(None, n_output))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        # 新序列预测时需要的encoder
        encoder_model = Model(encoder_inputs, encoder_states)
        # 新序列预测时需要的decoder
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                         initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        return model, encoder_model, decoder_model

    def train(self, array_x, array_y, array_yin, model_sub_dir):
        """训练模型并保存模型文件
        Args:
          array_x: encoder 的输入序列
          array_y: deocder 的输出序列
          array_yin: decoder 的 inference 序列
          model_sub_dir: 存放此次训练所生成的模型的目录
        """
        logger.info(f"Current value: 滞后期={self.n_in}, pca={self.pca_n}")

        n_input = array_x.shape[2]
        n_output = array_y.shape[2]
        model, encoder, decoder = self.build_models(n_input, n_output)
        model.fit([array_x, array_yin], array_y, batch_size=self.batch_size, epochs=self.epochs, verbose=2)
        encoder_path = cat_path(model_sub_dir, 'encoder.h5')
        decoder_path = cat_path(model_sub_dir, 'decoder.h5')
        encoder.save(encoder_path)
        decoder.save(decoder_path)
