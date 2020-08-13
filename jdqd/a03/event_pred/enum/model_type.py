# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改。模型状态枚举类，用于标识使用的模型类型。
class ModelType(Enum):
    RNN = '1'      # RNN
    CNN = '2'        # CNN
