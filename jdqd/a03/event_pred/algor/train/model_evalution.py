# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np
import jdqd.a03.event_pred.algor.common.pgsql_util as pgsql
from feedwork.utils import logger
from jdqd.a03.event_pred.enum.model_type import ModelType
from jdqd.a03.event_pred.algor.common import preprocess as pp
from jdqd.a03.event_pred.algor.common.model_util import load_rnn_models, load_cnn_model
from jdqd.a03.event_pred.algor.predict import predict_rnn, predict_cnn


def aggre_preds(preds):
    """
    将同一事件的多天预测值按照取最大值进行合并, 即, 如果一天一个事件类别有多次预测的多个预测
    值, 则保留该类别最大的预测值.
    Args:
      preds: 预测结果, shape(样本数, 预测天数, 事件类别数)

    Returns:
      合并后的预测结果, shape(样本数 + 预测天数 - 1, 事件类别数)

    """
    n_samples, n_days, n_classes = preds.shape
    canvas = np.zeros([n_samples, n_samples + n_days - 1, n_classes])
    # 将每个样本的预测值按照日期先后放到 canvas 上, 并使多个样本的预测日期在 canvas 第二个维度
    # 上对齐
    for i, p in enumerate(preds):
        canvas[i, i: i + n_days] = p
    preds_aggre = np.max(canvas, axis=0)
    return preds_aggre


def to_binary(preds, event_col):
    """
    将预测值按照指定事件类别转换成二分类(0, 1)形式
    Args:
      preds: 预测值, shape(样本数, 预测天数, 类别数)
      event_col: 事件在预测结果中的列数

    Returns:
      二分类形式的预测值, shape(样本数, 预测天数)
    """
    # 获取预测事件所在列
    shape = preds.shape
    if shape[-1] == 1:
        preds = (preds >= 0.5).astype(int)
        preds = preds.reshape(shape[:2]) # shape(样本数, 预测天数)
    else:
        preds = np.argmax(preds, axis=-1)
        preds = (preds == event_col).astype(int)
    return preds


def buffered_eval(preds, label_flatten, days_buffer=2):
    """
        计算带缓冲区的fp, tp 以及真实值正例负例个数
        Args:
          preds: 二分类的预测结果, shape(样本数, 预测天数)
          label_flatten: 预测结果对应的真实标签, 同为二分类表示

        Returns:
          tp, fp, 真实值正例个数, 真实值负例个数
        """
    neg_idxes = [i for i, l in enumerate(label_flatten) if l != 1]
    pos_idxes = [i for i, l in enumerate(label_flatten) if l == 1]
    pos_idxes_buffered = {p: list(range(p - days_buffer, p + 1)) for
                          p in pos_idxes}
    false_pos_idxes = []
    # 基于预测值的tp
    true_pos_idxes = []
    # 基于真实值的tp
    true_pos_idxes_label = []
    for i, p in enumerate(preds):
        # 遍历预测天数
        for j, pd in enumerate(p):
            if pd == 0:
                continue
            # 正例下标, 第 i 个样本预测值的第 j 天, 下标为 i + j
            pos_idx = i + j
            pos_idx_in_buffered = False

            for pi, pib in pos_idxes_buffered.items():
                if pos_idx in pib:
                    true_pos_idxes.append(pos_idx)
                    true_pos_idxes_label.append(pi)
                    pos_idx_in_buffered = True
                    break
            if not pos_idx_in_buffered:
                false_pos_idxes.append(pos_idx)
    num_tp = len(set(true_pos_idxes))
    num_tp_label = len(set(true_pos_idxes_label))
    num_fp = len(set(false_pos_idxes))
    num_pos = len(pos_idxes)
    num_neg = len(neg_idxes)
    return num_tp, num_tp_label, num_fp, num_pos, num_neg


def cal_false_alert(preds_flatten, outputs_flatten):
    """
    计算虚警率, fp / 预测与真实值不重复正例个数
    :param preds_flatten:
    :param outputs_flatten:
    :return: 虚警率, fp 数量, 预测值与真实值不重复正例数
    """
    comb = preds_flatten + outputs_flatten
    comb = (comb > 0).astype(int)
    false = ((preds_flatten - outputs_flatten) == 1).astype(int)
    num_fa = np.sum(false)
    num_comb_pos = np.sum(comb)
    return round(num_fa / num_comb_pos, 4), num_fa, num_comb_pos


def pred_rank(preds_aggre, label_flatten, top_num, event_col, days_buffer=2):
    """
    对指定事件的预测值进行排序, 计算排名前top_num位中预测正确的比率, 以及预测正确的个数在所有真正例中的比率
    :param top_num:
    :param preds_aggre: shape(样本数, 预测天数)
    :param label_flatten: shape(样本数, 1)
    :param days_buffer: 缓冲天数
    :return:
    """
    sort_idxes = preds_aggre[:, event_col].argsort()
    top_idxes = sort_idxes[-top_num:]
    true_in_top = 0
    for ti in top_idxes:
        if np.sum(label_flatten[
                  max(0, ti - days_buffer): ti + 1]) > 0:
            true_in_top += 1
    true_positive = len(np.where(label_flatten > 0)[0])
    return true_in_top / top_num, true_in_top / true_positive


def bleu(candidate, reference):
    """
    计算bleu值, 即candidate中指定长度的切片在reference中出现的次数与切片个数的比值.
      计算切片长度为1, 2, 3, 4 的 bleu 值, 再求均值
    Args:
      candidate: 候选序列, 即预测值
      reference: 参考序列, 即真实值

    Returns: 切片长度分别为1, 2, 3, 4 的 bleu 值均值
    """
    scores = []
    for i in [1, 2, 3, 4]:
        s_cadi_ap = list()
        s_refer_ap = list()
        s_cadi_dic = dict()
        s_refer_dic = dict()
        accur_count = 0
        for j in range(0, len(candidate) - i + 1):
            s_cadi = candidate[j:j + i]
            s_cadi_ap.append(str(s_cadi))
            s_refer = reference[j:j + i]
            s_refer_ap.append(str(s_refer))
        for k in s_cadi_ap:
            s_cadi_dic[k] = s_cadi_dic.get(k, 0) + 1

        for k in s_refer_ap:
            s_refer_dic[k] = s_refer_dic.get(k, 0) + 1

        for k in s_cadi_dic.keys():
            if k in s_refer_dic.keys():
                if s_cadi_dic[k] >= s_refer_dic[k]:
                    accur_count += s_refer_dic[k]
                else:
                    accur_count += s_cadi_dic[k]

        score = round(accur_count / len(s_cadi_ap), 2)
        scores.append(score)
    avg_score = np.average(scores)
    return avg_score


def evaluate_sub_model_by_event(event_col, preds, outputs_test):
    """根据特定事件评估子模型
    Args:
      event_col: 事件在补 0 one-hot 形式事件列表中的列号
      preds: 预测结果, shape(样本数, 预测天数, 事件类别个数)
      outputs_test: 预测结果对应的真实标签, shape(样本数, 预测天数, 事件类别个数)

    Returns:
      子模型的评估值
    """
    preds_aggre = aggre_preds(preds)
    preds = to_binary(preds, event_col)  # shape(样本数, 预测天数)
    preds_shape = preds.shape # @todo 改成使用flatten_outputs
    preds_ = np.reshape(preds, [*preds_shape, 1])  # shape(样本数, 预测天数, 1)
    preds_flatten = aggre_preds(preds_)  # shape(样本数 + output_len - 1, 1)
    preds_flatten = preds_flatten.reshape([-1]).astype(int)  # shape(样本数 + output_len - 1,)

    outputs_test_ = to_binary(outputs_test, event_col) # shape(样本数, 预测天数)

    outputs_test_ = pp.flatten_outputs(outputs_test_)  # shape(样本数 + output_len - 1,)
    tier_precision, tier_recall = pred_rank(preds_aggre, outputs_test_, 10,
                                            event_col)
    bleu_score = bleu(preds_flatten, outputs_test_)

    num_tp, num_tp_label, num_fp, num_pos, num_neg = \
        buffered_eval(preds_, outputs_test_)

    false_alert, num_fa, num_comb_pos = \
        cal_false_alert(preds_flatten, outputs_test_)
    return tier_precision, tier_recall, bleu_score, false_alert, num_fp, \
           num_tp, num_tp_label, num_neg, num_pos, num_fa, num_comb_pos


def evaluate_sub_model(preds, outputs_test, events_set, events_num):
    """
    评估子模型.
    :param preds: 预测结果, shape(样本数, 预测天数, 事件类别个数)
    :param outputs_test: 预测结果对应的真实值, shape 同 preds
    :param events_set: 去重后的全部数据集中的事件类别的有序列表
    :param events_num: dict, key为事件类别, value 为事件在评估日期内出现的次数
    :return: 子模型的评估结果, 分别返回多个事件的综合评估值(evals_summary), 针对各个事件的
    评估值(evals_separate), 以及被评估的事件列表(eval_events)
    """
    evals_separate = []
    eval_events = []
    bleus = []
    tier_precisions = []
    tier_recalls = []
    num_fps = []
    num_negs = []
    num_tps = []
    num_tp_labels = []
    num_poses = []
    num_fas = []
    num_comb_poses = []
    for i, event in enumerate(events_set):
        if str(event) == '0':
            continue
        event_num = events_num[event]
        if event_num == 0:
            continue
        tier_precision, tier_recall, bleu_score, false_alert, num_fp, num_tp,  num_tp_label, num_neg, num_pos, num_fa, \
            num_comb_pos = evaluate_sub_model_by_event(i, preds, outputs_test)

        tier_precision = round(tier_precision, 4)
        tier_recall = round(tier_recall, 4)
        bleu_score = round(bleu_score, 4)
        false_alert = round(false_alert, 4)

        false_report_denom = num_tp + num_fp
        if false_report_denom == 0:
            false_report = 0
        else:
            false_report = round(num_fp / false_report_denom, 4)
        recall = round(num_tp_label / num_pos, 4)

        num_fps.append(num_fp)
        num_negs.append(num_neg)

        num_tps.append(num_tp)
        num_tp_labels.append(num_tp_label)
        num_poses.append(num_pos)

        num_fas.append(num_fa)
        num_comb_poses.append(num_comb_pos)

        tier_precisions.append(tier_precision)
        tier_recalls.append(tier_recall)
        bleus.append(bleu_score)
        evals_of_event = [event, event_num, false_report, recall, false_alert,
                          tier_precision, tier_recall, bleu_score]
        evals_separate.append(evals_of_event)
        eval_events.append(event)
    evals_summary = []
    if bleus:
        bleu_summary = round(np.mean(bleus), 4)
        tier_precision_summary = round(np.mean(tier_precisions), 4)
        tier_recall_summary = round(np.mean(tier_recalls), 4)
        rc_summary = round(np.sum(num_tp_labels) / np.sum(num_poses), 4)
        fr_summary_denom = np.sum(num_tps + num_fps)
        if fr_summary_denom == 0:
            fr_summary = 0
        else:
            fr_summary = round(np.sum(num_fps) / fr_summary_denom, 4)
        fa_summary = round(np.sum(num_fas) / np.sum(num_comb_poses), 4)
        # score_summary = 2 * (1 - fr_summary) * rc_summary / ((1 - fr_summary) + rc_summary)
        score_summary = (1 - fr_summary) * rc_summary
        score_summary = round(score_summary, 4)
        evals_summary = [score_summary, bleu_summary, tier_precision_summary,
                         tier_recall_summary, fr_summary, rc_summary,
                         fa_summary]

    return evals_summary, evals_separate, eval_events


def evaluate_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set, n_classes,
                    start_date, end_date, model_type, eval_event, model_dir):
    """
    评估子模型并将评估结果存入数据库
    Args:
      data: 评估范围内的数据表数据, 用于生成预测值
      dates: 数据表的日期列表, 全部日期, 不仅为评估范围内
      detail_ids: 子模型对应的 detail_id 列表
      sub_model_dirs: 子模型对应的模型文件存放路径列表
      params_list: 子模型对应的超参数列表
      events_p_oh: 补 0 one-hot 形式的事件列表, 全部日期范围
      events_set: 去重后有序事件集合
      n_classes: 事件类别个数, 包含 0 事件, 为预测值最后一维长度
      start_date: 开始评估日期, 为预测开始日期
      end_date: 结束评估日期, 为预测结束日期
      model_type: string.模型类型，对应代码项为ModelType
      eval_event: cnn模型专用参数，该参数不为none时表示该模型是cnn
      model_dir: str，模型存放地址

    Returns:
      scores: 子模型综合评分列表
      events_num: dict, 测试集事件出现次数: {event: event_num}
    """
    scores = []
    events_num = {}

    for detail_id, sub_model_dir, params in zip(detail_ids, sub_model_dirs, params_list):
        logger.info(f'评估模型: {sub_model_dir}, detail_id: {detail_id}')
        if ModelType.CNN.value == model_type:
            input_len, output_len = params

            inputs_test, outputs_test = pp.gen_samples_by_pred_date(data, events_p_oh, input_len, output_len, dates,
                                                                    start_date, end_date)
            event_col = events_set.index(eval_event)
            outputs_test = outputs_test[:, :, event_col]
            outputs_test = np.reshape(outputs_test, [*outputs_test.shape, 1])
            events_set_ = [eval_event]

            model = load_cnn_model(sub_model_dir)
            preds = predict_cnn.predict_samples(model, inputs_test)
        elif ModelType.RNN.value == model_type:
            input_len, output_len, n_pca = params
            values_pca = pp.apply_pca(n_pca, model_dir, data, True)
            events_set_ = events_set

            inputs_test, outputs_test = \
                pp.gen_samples_by_pred_date(values_pca, events_p_oh, input_len, output_len, dates, start_date, end_date)
            encoder, decoder = load_rnn_models(sub_model_dir)
            preds = predict_rnn.predict_samples(encoder, decoder, inputs_test, output_len, n_classes)
        else:
            raise RuntimeError(f"Unsupport model type {model_type}")

        events_num = pp.get_event_num(outputs_test, events_set_)
        evals_summary, evals_separate, eval_events = evaluate_sub_model(preds, outputs_test, events_set_, events_num)
        if len(evals_summary) == 0:     # 若evals_summary无元素，则表示评估的事件在评估日期范围内没有发生
            return None, None

        for event, eval_separate in zip(eval_events, evals_separate):
            event, event_num, false_report, recall, false_alert, tier_precision, tier_recall, bleu_score = eval_separate
            event_num = events_num[event]
            pgsql.insert_model_test(event, event_num, false_report, recall, false_alert, tier_precision, tier_recall,
                                    bleu_score, detail_id)

        score_summary, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary, fa_summary = \
            evals_summary

        scores.append([score_summary, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary,
                       fa_summary, detail_id])

    return scores, events_num


