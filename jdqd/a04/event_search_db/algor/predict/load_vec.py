#!/usr/bin/env python
# coding:utf-8
# 读取向量文件
import traceback
import numpy as np
import json
from feedwork.utils import logger


def load_vec_data(db, cameo=""):
    """
    传入事件cameo号，到cameo号对应的列表中加载所有事件短句向量
    :param cameo:(str)事件cameo号
    """
    # 游标对象
    cursor = db.cursor()
    # 定义好sql语句，%s是字符串的占位符
    if cameo and cameo is not None:
        sql = f"select * from event_vec_table where cameo='{cameo}'"
    else:
        sql = "select * from event_vec_table"
    # 数据向量字典
    data = {}
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 获取所有结果集
        results = cursor.fetchall()
        for once in results:
            # 将读取到的字节流转化为ndarray数组
            try:
                numArr = np.array(json.loads(bytes.decode(bytes(once[2], encoding="utf-8"))), dtype=np.float32)
            except:
                numArr = np.array(json.loads(bytes.decode(bytes(once[2], encoding="gbk"))), dtype=np.float32)
            data[once[1]] = numArr
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        raise trace

    finally:
        cursor.close()

    return data


def load_cameo_dict(db):
    """
    传入数据库连接，获取数据库中的向量，获取其中所有的cameo,id用于构造cameo2id字典
    :return:data(dict){cameo:[event_id]}
    """
    # 游标对象
    cursor = db.cursor()
    # 定义好sql语句，%s是字符串的占位符
    sql = "select * from event_vec_table"
    # 数据向量字典
    data = {}
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 获取所有结果集
        results = cursor.fetchall()
        for once in results:
            event_ids = data.setdefault(once[0], [])
            event_ids.append(once[1])
    except:
        trace = traceback.format_exc()
        logger.error(trace)
        raise trace
    finally:
        cursor.close()

    return data


if __name__ == "__main__":
    db = ""
    x = np.load("eventmerge/resources/vec_data/000.npy")
    main_vec = x[0]
    vec = load_vec_data(db)

    print(main_vec)
