# -*- coding:utf-8 -*-
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改
class ModelStatus(Enum):
    SUCCESS = '2'      # 成功
    PROCESSING = "1"   # 进行中
    FAILD = '3'        # 失败
