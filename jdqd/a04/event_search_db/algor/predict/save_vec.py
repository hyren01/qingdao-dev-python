# coding:utf-8
# 将向量保存到数据库
import json
import traceback
import numpy as np
from feedwork.utils import logger


def save_vec_data(db, cameo, event_id, main_vec):
    """
    将事件cameo, event_id, main_vec保存到数据库中
    :param cameo: (str)事件cameo号
    :param event_id: (str)事件编号
    :param main_vec: (ndarray)事件向量
    :return: None
    """
    if not isinstance(cameo, str) or not cameo:
        logger.error("cameo编号格式错误!")
        raise TypeError
    if not isinstance(event_id, str) or not event_id:
        logger.error("事件编号格式错误!")
        raise TypeError

    # 将向量转化为json字符串格式
    vector = json.dumps(main_vec.tolist())
    # 创建游标
    cursor = db.cursor()
    # 插入语句
    sql = "insert into event_vec_table(cameo,event_id,event_vec) values(%s,%s,%s)"

    try:
        # 执行sql语句
        cursor.execute(sql, (cameo, event_id, vector))
        # 提交到数据库中
        db.commit()
    except:
        db.rollback()
        trace = traceback.format_exc()
        logger.error(trace)
        raise trace
    finally:
        cursor.close()


if __name__=="__main__":
    db = ""
    x = np.load("eventmerge/resources/vec_data/000.npy")
    main_vec = x[0]
    save_vec_data(db, "0", "0", main_vec)