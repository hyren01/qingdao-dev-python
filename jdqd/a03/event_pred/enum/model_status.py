# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改。模型状态枚举类，用于标识模型训练及预测时的状态。
class ModelStatus(Enum):
    PROCESSING = "1"   # 进行中
    SUCCESS = '2'      # 成功
    FAILD = '3'        # 失败
