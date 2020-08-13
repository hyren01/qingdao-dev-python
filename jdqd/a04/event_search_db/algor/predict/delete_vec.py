#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月13
import traceback
from feedwork.utils import logger


def execute_delete(db, event_id):
    """
    删除模块的主控程序，读取cameo2id,然后查看事件id是否存在字典中，并进行删除。
    :return: None
    :raise: FileNotFoundError
    """
    cursor = db.cursor()
    # 游标对象
    sql = "delete from event_vec_table where event_id=(%s)"
    # 定义好sql语句，%s是字符串的占位符

    try:
        cursor.execute(sql, (event_id))
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
    execute_delete(db, "0")