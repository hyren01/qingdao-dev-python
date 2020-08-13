#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mr Fan
# @Time: 2020年05月13
import psycopg2
from jdqd.a04.event_search_db.config import PredictConfig


def get_connection():
    """
    创建数据库连接
    :return: connect
    """
    # 数据库连接
    connect = psycopg2.connect(
        host=PredictConfig.db_host,
        port=PredictConfig.db_port,
        database=PredictConfig.db_name,
        user=PredictConfig.db_user,
        password=PredictConfig.db_passwd
    )

    return connect
