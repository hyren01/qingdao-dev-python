# -*- coding:utf-8 -*-
from enum import Enum, unique


# 如果不继承Enum，枚举值能被修改
class EventType(Enum):
    SUPER_EVENT_TYPE = '2'      # 大类事件
    SUB_EVENT_TYPE = '1'        # 小类事件
