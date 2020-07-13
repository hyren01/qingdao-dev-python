# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改。事件类别类型枚举类
class EventType(Enum):
    SUPER_EVENT_TYPE = '1'      # 大类事件
    SUB_EVENT_TYPE = '2'        # 小类事件
