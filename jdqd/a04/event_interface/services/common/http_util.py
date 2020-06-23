#!/usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.parse import urlencode
from urllib.request import urlopen


def http_post(data, uri):

    data = urlencode(data)
    data = data.encode()
    res = urlopen(url=uri, data=data)
    content = res.read()

    return content
