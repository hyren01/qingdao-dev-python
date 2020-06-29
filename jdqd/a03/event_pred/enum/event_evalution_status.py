# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改
class ModelEvalutionStatus(Enum):
    EFFECTIVE = '1'      # 有效
    INVALID = "0"        # 无效
