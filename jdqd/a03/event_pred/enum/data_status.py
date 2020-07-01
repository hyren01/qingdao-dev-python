# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改。数据状态枚举类，用于标识数据状态
class DataStatus(Enum):
    SUCCESS = '1'      # 成功/有效
    FAILD = '0'        # 失败/无效
