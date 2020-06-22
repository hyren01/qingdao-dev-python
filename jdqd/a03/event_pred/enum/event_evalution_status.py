# -*- coding:utf-8 -*-
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改
class ModelEvalutionStatus(Enum):
    EFFECTIVE = '1'      # 有效
    INVALID = "0"        # 无效
