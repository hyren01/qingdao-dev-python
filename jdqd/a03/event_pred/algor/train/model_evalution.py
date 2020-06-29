# -*- coding: utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
import numpy as np
import jdqd.a03.event_pred.algor.common.pgsql_util as pgsql
import feedwork.AppinfoConf as appconf
from feedwork.utils import logger
from jdqd.a03.event_pred.algor.common import preprocess as pp
from jdqd.a03.event_pred.algor.predict import predict
from feedwork.utils.FileHelper import cat_path
from jdqd.a03.event_pred.enum.event_evalution_status import ModelEvalutionStatus

model_dir = cat_path(appconf.ALGOR_MODULE_ROOT, 'event_pred')


# @todo(zhxin): 数据维度含义明细

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
    pos_idxes_buffered = {p: list(range(p - days_buffer, p + days_buffer + 1)) for
                          p in pos_idxes}
    false_pos_idxes = []
    true_pos_idxes = []
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
                    pos_idx_in_buffered = True
                    break
            if not pos_idx_in_buffered:
                false_pos_idxes.append(pos_idx)
    num_tp = len(set(true_pos_idxes))
    num_fp = len(set(false_pos_idxes))
    num_pos = len(pos_idxes)
    num_neg = len(neg_idxes)
    return num_tp, num_fp, num_pos, num_neg


def buffered_eval2(preds, label_flatten, days_buffer=2):
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
    pos_idxes_buffered = {p: list(range(p - days_buffer, p + days_buffer + 1)) for
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
    :param preds: shape(样本数, 预测天数)
    :param label_flatten: shape(样本数, 1)
    :param days_buffer: 缓冲天数
    :return:
    """
    sort_idxes = preds_aggre[:, event_col].argsort()
    top_idxes = sort_idxes[-top_num:]
    true_in_top = 0
    for ti in top_idxes:
        if np.sum(label_flatten[max(0, ti - days_buffer): min(len(label_flatten) - 1, ti + days_buffer)]) > 0:
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
      preds: 预测结果
      outputs_test: 预测结果对应的真实标签

    Returns:
      子模型的评估值
    """
    preds_aggre = aggre_preds(preds)
    preds = to_binary(preds, event_col)
    preds_shape = preds.shape
    preds_ = np.reshape(preds, [*preds_shape, 1])
    preds_flatten = aggre_preds(preds_)
    preds_flatten = preds_flatten.reshape([-1]).astype(int)
    outputs_test_ = to_binary(outputs_test, event_col)
    if len(outputs_test_.shape) > 1:
        outputs_test_ = pp.flatten_outputs(outputs_test_)
    if len(preds_flatten) > len(outputs_test_):
        preds_flatten = preds_flatten[:len(outputs_test_)]
    tier_precision, tier_recall = pred_rank(preds_aggre, outputs_test_, 10,
                                            event_col)
    bleu_score = bleu(preds_flatten, outputs_test_)

    num_tp, num_tp_label, num_fp, num_pos, num_neg = buffered_eval2(preds_, outputs_test_)

    false_alert, num_fa, num_comb_pos = cal_false_alert(preds_flatten, outputs_test_)
    return tier_precision, tier_recall, bleu_score, false_alert, num_fp, num_tp, num_tp_label, num_neg, num_pos, \
           num_fa, num_comb_pos


def evaluate_sub_models(data, dates, detail_ids, sub_model_dirs, params_list, events_p_oh, events_set, n_classes,
                        start_date, end_date):
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

    Returns:
      scores: 子模型综合评分列表
      events_num: dict, 测试集事件出现次数: {event: event_num}
    """
    scores = []
    events_num = {}

    for detail_id, sub_model_dir, params in zip(detail_ids, sub_model_dirs, params_list):
        logger.info(f'评估模型: {sub_model_dir}, detail_id: {detail_id}')
        input_len, output_len, n_pca = params
        preds, outputs_test = pred(sub_model_dir, data, dates, events_p_oh, input_len, output_len, n_classes, n_pca,
                                   start_date, end_date)
        events_num = pp.get_event_num(outputs_test, events_set)
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
        # TODO 各种写死的值至少要写出表示什么意思
        for i, event in enumerate(events_set):
            if str(event) == '0':
                continue
            event_num = events_num[event]
            if event_num == 0:
                continue
            tier_precision, tier_recall, bleu_score, false_alert, num_fp, num_tp, num_tp_label, num_neg, num_pos, \
                num_fa, num_comb_pos = evaluate_sub_model_by_event(i, preds, outputs_test)

            tier_precision = round(tier_precision, 4)
            tier_recall = round(tier_recall, 4)
            bleu_score = round(bleu_score, 4)
            false_alert = round(false_alert, 4)

            false_report_denom = num_fp + num_fp
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
            # TODO 不同目录下的py文件都有数据库操作，应该规划好把这些不同性质的操作分离，这个地方没有记录评估失败的情况
            #   应该有个方法为"模型预测"，多个子模型的循环应该由调用该方法的外层来完成，并且数据库操作统一由外层完成
            pgsql.insert_model_test(event, event_num, false_report, recall, false_alert, tier_precision, tier_recall,
                                    bleu_score, ModelEvalutionStatus.EFFECTIVE.value, detail_id)

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
            score = (1 - fr_summary) * rc_summary
            score = round(score, 4)
            scores.append([score, bleu_summary, tier_precision_summary, tier_recall_summary, fr_summary, rc_summary,
                           fa_summary, detail_id])

    return scores, events_num


def pred(sub_model_dir, data, dates, events_p_oh, input_len, output_len, n_classes, n_pca, start_date, end_date):
    """
    预测评估日期内的数据
    Args:
      sub_model_dir: 子模型文件存放目录
      data: 评估范围内的数据表数据, 用于生成预测值
      dates: 数据表的日期列表, 全部日期, 不仅为评估范围内
      events_p_oh: 补 0 one-hot 形式的事件列表, 全部日期范围
      input_len: encoder 输入序列长度, 即预测所需数据天数
      output_len: decoder 输出序列长度, 即预测天数
      n_classes: 事件类别个数, 包含 0 事件, 为预测值最后一维长度
      start_date: 开始评估日期, 为预测开始日期
      end_date: 结束评估日期, 为预测结束日期
      n_pca: 降维维度数

    Returns:
      评估日期范围内的预测值及真实标签值
    """
    values_pca = pp.apply_pca(n_pca, model_dir, data, True)

    inputs_test, outputs_test = pp.gen_samples_by_pred_date(values_pca, events_p_oh, input_len, output_len, dates,
                                                            start_date, end_date)
    preds = predict.predict(sub_model_dir, inputs_test, output_len, n_classes)
    preds = np.array(preds)

    return preds, outputs_test
