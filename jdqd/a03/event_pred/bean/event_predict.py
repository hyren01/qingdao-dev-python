# -*- coding:utf-8 -*-
"""
实体类，用于封装预测时用的信息，不对应数据库表。
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""


class EventPredict(object):

    def __init__(self):
        self._model_name = ''
        self._detail_id = ''
        self._lag_date = None
        self._pca = None
        self._days = None
        self._kernel_size = None
        self._pool_size = None
        self._model_dir = ''

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def detail_id(self):
        return self._detail_id

    @detail_id.setter
    def detail_id(self, detail_id: str):
        self._detail_id = detail_id

    @property
    def lag_date(self):
        return self._lag_date

    @lag_date.setter
    def lag_date(self, lag_date: int):
        self._lag_date = lag_date

    @property
    def pca(self):
        return self._pca

    @pca.setter
    def pca(self, pca: int):
        self._pca = pca

    @property
    def days(self):
        return self._days

    @days.setter
    def days(self, days: int):
        self._days = days

    @property
    def kernel_size(self):
        return self._days

    @kernel_size.setter
    def kernel_size(self, kernel_size: int):
        self._kernel_size = kernel_size

    @property
    def pool_size(self):
        return self._pool_size

    @pool_size.setter
    def pool_size(self, pool_size: int):
        self._pool_size = pool_size

    @property
    def model_dir(self):
        return self._model_dir

    @model_dir.setter
    def model_dir(self, model_dir):
        self._model_dir = model_dir
