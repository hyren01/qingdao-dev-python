#!/usr/bin/env python
# -*- coding:utf-8 -*-
import hashlib
import time
import uuid


def gener_id_by_tiemstamp():
    time_str = str(round(time.time() * 1000))
    m = hashlib.md5()
    m.update(time_str.encode("utf8"))
    md5 = m.hexdigest()
    return md5


# 基于时间戳。由MAC地址、当期时间戳、随机数生成。保证全球范围内的唯一性
def gener_id_by_uuid():
    id = str(uuid.uuid1()).replace('-', '')
    return id


if __name__ == '__main__':
    print(gener_id_by_uuid())