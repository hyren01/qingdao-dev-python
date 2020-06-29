# -*- coding:utf-8 -*-
"""
@Author: zhang xin
@Time: 2020/6/16 10:38
desc:
"""


class EventModel(object):

    def __init__(self):
        self._model_id = ''
        self._model_name = ''
        self._tables_name = ''
        self._dr_min = None
        self._dr_max = None
        self._delay_min_day = None
        self._delay_max_day = None
        self._neure_num = None
        self._train_batch_no = None
        self._epoch = None
        self._days = None
        self._tran_start_date = ''
        self._tran_end_date = ''
        self._evaluation_start_date = ''
        self._evaluation_end_date = None
        self._size = None
        self._status = None
        self._create_date = None
        self._create_time = None
        self._event_type = None
        self._tran_finish_date = ''
        self._tran_finish_time = ''

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def tables_name(self):
        return self._tables_name

    @tables_name.setter
    def tables_name(self, tables_name: bool):
        self._tables_name = tables_name

    @property
    def dr_min(self):
        return self._dr_min

    @dr_min.setter
    def dr_min(self, dr_min):
        self._dr_min = dr_min

    @property
    def dr_max(self):
        return self._dr_max

    @dr_max.setter
    def dr_max(self, dr_max):
        self._dr_max = dr_max

    @property
    def delay_min_day(self):
        return self._delay_min_day

    @delay_min_day.setter
    def delay_min_day(self, delay_min_day):
        self._delay_min_day = delay_min_day

    @property
    def delay_max_day(self):
        return self._delay_max_day

    @delay_max_day.setter
    def delay_max_day(self, delay_max_day):
        self._delay_max_day = delay_max_day

    @property
    def neure_num(self):
        return self._neure_num

    @neure_num.setter
    def neure_num(self, neure_num):
        self._neure_num = neure_num

    @property
    def train_batch_no(self):
        return self._train_batch_no

    @train_batch_no.setter
    def train_batch_no(self, train_batch_no):
        self._train_batch_no = train_batch_no

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def days(self):
        return self._days

    @days.setter
    def days(self, days):
        self._days = days

    @property
    def tran_start_date(self):
        return self._tran_start_date

    @tran_start_date.setter
    def tran_start_date(self, tran_start_date):
        self._tran_start_date = tran_start_date

    @property
    def tran_end_date(self):
        return self._tran_end_date

    @tran_end_date.setter
    def tran_end_date(self, tran_end_date):
        self._tran_end_date = tran_end_date

    @property
    def evaluation_start_date(self):
        return self._evaluation_start_date

    @evaluation_start_date.setter
    def evaluation_start_date(self, evaluation_start_date):
        self._evaluation_start_date = evaluation_start_date

    @property
    def evaluation_end_date(self):
        return self._evaluation_end_date

    @evaluation_end_date.setter
    def evaluation_end_date(self, evaluation_end_date):
        self._evaluation_end_date = evaluation_end_date

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def create_date(self):
        return self._create_date

    @create_date.setter
    def create_date(self, create_date):
        self._create_date = create_date

    @property
    def create_time(self):
        return self._create_time

    @create_time.setter
    def create_time(self, create_time):
        self._create_time = create_time

    @property
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, event_type):
        self._event_type = event_type

    @property
    def tran_finish_date(self):
        return self._tran_finish_date

    @tran_finish_date.setter
    def tran_finish_date(self, tran_finish_date):
        self._tran_finish_date = tran_finish_date

    @property
    def tran_finish_time(self):
        return self._tran_finish_time

    @tran_finish_time.setter
    def tran_finish_time(self, tran_finish_time):
        self._tran_finish_time = tran_finish_time
